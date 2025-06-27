import os
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.BrainDataset_2D import TestData
from utils_2d.warp import Warper2d, warp2D
from modal_2d.RegFusion_lite import Encoder, ModelTransfer_lite, RegNet_lite, FusionNet_lite
import torch

from utils_2d.utils import project, rgb2ycbcr, ycbcr2rgb
import warnings

# ----------------- 定义函数和类 (这些可以放在 if __name__ == '__main__': 块外面) -----------------

# 忽略警告，虽然通常不推荐，但如果你确定这些警告是“可接受的”
warnings.filterwarnings('ignore')


def validate_mask(encoder, transfer, reg_net, fusion_net, dataloader, modal):
    # torch.cuda.empty_cache() # 如果使用GPU，可以取消注释
    # torch.backends.cudnn.benchmark = True # 如果使用GPU，可以取消注释
    epoch_iterator = tqdm(dataloader, desc='Val (X / X Steps) (loss=X.X)', ncols=150, leave=True, position=0)
    encoder.eval()
    transfer.eval()
    reg_net.eval()
    fusion_net.eval()

    figure_save_path = f"./{modal}_result"
    if not os.path.exists(figure_save_path):
        os.makedirs(os.path.join(figure_save_path, "MRI"))
        os.makedirs(os.path.join(figure_save_path, f"{modal}"))
        os.makedirs(os.path.join(figure_save_path, f"Fusion"))
        os.makedirs(os.path.join(figure_save_path, f"{modal}_align"))
        os.makedirs(os.path.join(figure_save_path, f"{modal}_label"))

    with torch.no_grad():
        for i, batch in enumerate(epoch_iterator):
            img1, img2, file_name = batch
            H, W = img1.shape[2], img1.shape[3]

            # 确保设备正确，如果使用多进程，这里最好放在 try-except 块中，以捕获worker内部的错误
            # 并且，注意 device 如果是 "cuda"，确保有多张GPU时不会冲突
            img1, img2 = img1.to(device), img2.to(device)

            if modal != 'CT':  # 只有在modal不是CT时才进行RGB转YCbCr
                img2_ycbcr = rgb2ycbcr(img2)  # 先保存完整的YCbCr版本，因为img2后面会被替换
                img2_cbcr = img2_ycbcr[:, 1:3, :, :]
                img2 = img2_ycbcr[:, 0:1, :, :]  # img2 现在是 Y 通道

            AS_F, feature1 = encoder(img1)
            BS_F, feature2 = encoder(img2)  # 注意这里 img2 可能是Y通道
            pre1, pre2, feature_pred1, feature_pred2, feature1, feature2, AU_F, BU_F = transfer(feature1, feature2)

            feature1 = project(feature1, [H, W]).to(device)
            feature2 = project(feature2, [H, W]).to(device)
            AU_F = project(AU_F, [H, W])
            BU_F = project(BU_F, [H, W])

            # img1, img2 = img1.to(device), img2.to(device) # 这两行是多余的，前面已经to(device)了

            _, _, flows, flow, _, _ = reg_net(feature1, feature2)
            warped_image2 = image_warp(flow, img2)  # warped_image2 也是Y通道
            fusion_img = fusion_net(AS_F, BS_F, AU_F, BU_F, flow)  # fusion_img 也是Y通道

            """save as png"""
            if modal == 'CT':
                # CT 模式下，图像是单通道，直接保存
                pass  # 原代码这里是 pass，说明 CT 模式下 fusion_img 已经是最终形式
            else:
                # PET/其他模式下，将融合的Y通道与CbCr通道合并，并转回RGB
                fusion_cbcr = warp2D()(img2_cbcr, flow)  # 将原始 img2 的 CbCr 也进行配准
                fusion_img = torch.cat((fusion_img, fusion_cbcr), dim=1)  # 合并 Y 和配准后的 CbCr
                fusion_img = ycbcr2rgb(fusion_img)  # 转换为 RGB

            # 确保保存的图像在 [0, 1] 范围内或者根据 save_image 的要求进行归一化
            # save_image 会自动处理，只要数据类型正确即可
            save_image(img1.cpu(), os.path.join(figure_save_path, f"MRI/{file_name[0]}"))
            # 注意：img2 在非CT模式下是Y通道，这里保存的是Y通道图像，如果想保存原始RGB，需要调整
            # 这里的 img2 已经是处理后的Y通道，如果想保存原始输入，需要在前面保留一份原始img2
            save_image(img2.cpu(), os.path.join(figure_save_path, f"{modal}/{file_name[0]}"))
            save_image(fusion_img.cpu(), os.path.join(figure_save_path, f"Fusion/{file_name[0]}"))
            save_image(warped_image2.cpu(), os.path.join(figure_save_path, f"{modal}_align/{file_name[0]}"))

    return


# ----------------- 主执行逻辑 (必须放在 if __name__ == '__main__': 块中) -----------------
if __name__ == '__main__':
    # 启用 multiprocessing 的 freeze_support (在Windows上打包成exe时需要，平时运行脚本可省略)
    # from multiprocessing import freeze_support
    # freeze_support()

    modal = 'PET'  # 确保这里的 modal 与全局定义一致，或者作为参数传递

    image_warp = Warper2d()
    # 优先使用 CUDA 如果可用，否则使用 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    checkpoint_path = './checkpoint'

    # 加载模型检查点
    # 注意：如果模型是在GPU上训练的，并且现在要加载到CPU，map_location='cpu' 是必要的。
    # 如果现在有GPU且想用GPU，则可以不指定或指定 'cuda'
    checkpoint = torch.load(os.path.join(checkpoint_path, f'BSAFusion_{modal}.pkl'), map_location=device)

    # 初始化模型并加载权重
    encoder = Encoder().to(device)
    transfer = ModelTransfer_lite(num_vit=2, num_heads=4, img_size=[256, 256]).to(device)
    reg_net = RegNet_lite().to(device)
    fusion_net = FusionNet_lite().to(device)

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    transfer.load_state_dict(checkpoint['transfer_state_dict'])
    reg_net.load_state_dict(checkpoint['reg_net_state_dict'])
    fusion_net.load_state_dict(checkpoint['fusion_net_state_dict'])

    # 准备数据集和数据加载器
    if modal == 'CT':
        val_dataset = TestData(
            img1_folder=f'./data/testData/{modal}/MRI',
            img2_folder=f'./data/testData/{modal}/{modal}',
            modal=modal
        )
    else:
        val_dataset = TestData(
            img1_folder=f'./data/testData/{modal}/MRI',
            img2_folder=f'./data/testData/{modal}/{modal}_RGB',  # 假设PET图像是RGB的
            modal=modal
        )

    # num_workers 的设置：
    # 在 Windows 上，即使使用了 if __name__ == '__main__':，过高的 num_workers 仍然可能导致问题
    # 如果仍然报错，尝试将 num_workers 降低，甚至设置为 0 来排除多进程问题。
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        pin_memory=True,  # 如果使用GPU，pin_memory=True 有助于加速数据传输
        shuffle=False,
        num_workers=1  # 从1开始尝试，如果稳定再逐渐增加。在Windows上，可能0或1是比较稳妥的选择
    )

    # 调用主验证函数
    print("Starting validation...")
    validate_mask(encoder, transfer, reg_net, fusion_net, val_dataloader, modal)
    print("Validation finished.")