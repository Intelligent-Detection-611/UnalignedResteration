import os
import warnings
import torch
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import DataLoader
from colorama import Fore, Style

# 导入你自己的数据集 (确保路径正确)
from dataset.BrainDataset_2D import RegDataset_F

# 导入我们之前创建的独立配准网络
from modal_2d.registration_network import MedicalImageRegistrationNet

# 导入原始的损失函数 (我们将从中提取配准相关的部分)
# 确保 utils_2d/loss.py 文件存在且可导入
from utils_2d.loss import regFusion_loss

# 导入原始的 warp2D (尽管 MedicalImageRegistrationNet 内部已经处理了，但为了保持一致性)
from utils_2d.warp import warp2D

warnings.filterwarnings('ignore')
print(f'{Style.RESET_ALL}')

# 设置 GPU (根据你的机器配置修改)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 例如，如果你只有一块卡且想用第0块
# multiGPUs = False # 通常不需要显式设置，torch.nn.DataParallel 会处理
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 实例化 warp2D (尽管 MedicalImageRegistrationNet 内部已经处理了，这里可能不再直接使用)
# 但 RegDataset_F 可能需要，所以保留
image_warp = warp2D()


def train(modal,
          train_batch_size,
          lr,
          num_epoch,
          beta1,
          beta2,
          resume):
    checkpoint_dir = './registration_checkpoints'  # 更改检查点保存路径
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # 子文件夹用于保存每个epoch的检查点
    specific_checkpoint_path = os.path.join(checkpoint_dir, 'epoch_checkpoints')
    if not os.path.exists(specific_checkpoint_path):
        os.makedirs(specific_checkpoint_path)

    torch.backends.cudnn.benchmark = True  # 开启 cudnn 自动寻找最佳算法，加速训练

    # 数据集和数据加载器
    train_dataset = RegDataset_F(
        root='./data',  # 确保你的数据路径正确
        mode='train',
        model=modal,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        pin_memory=True,  # 如果内存允许且使用GPU，开启可以加速数据传输
        shuffle=True,
        num_workers=os.cpu_count() // 2 if os.cpu_count() is not None else 8  # 根据CPU核心数调整
    )

    """创建配准网络"""
    # 实例化我们独立的配准网络
    registration_net = MedicalImageRegistrationNet(img_size=[256, 256]).to(device)

    # 优化器
    # 只需要为 MedicalImageRegistrationNet 的所有参数创建一个优化器
    optimizer = Adam(registration_net.parameters(), lr=lr, betas=(beta1, beta2))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=lr * 1e-2)

    # 损失记录
    epoch_loss_values = []
    cls_loss_value = []
    transfer_loss_value = []
    flow_loss_value = []
    reg_loss_values = []
    # fusion_loss_value = [] # 融合损失在此配准训练中不需要

    start_epoch = 0
    # 这些 SSIM 相关的参数在原始损失函数中用于调整融合损失的权重
    # 在仅配准训练中，如果 regFusion_loss 内部没有其他地方用到，可以忽略其累加
    sum_ssim1 = 1.0
    sum_ssim2 = 1.0

    """恢复训练"""
    if resume:
        # 恢复检查点路径需要与保存逻辑匹配
        checkpoint_file = os.path.join(specific_checkpoint_path, 'registration_checkpoint_latest.pth')
        if os.path.exists(checkpoint_file):
            print(f"Resuming training from {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file, map_location=device)
            registration_net.load_state_dict(checkpoint['registration_net_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1  # 从下一轮开始
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # 恢复调度器状态

            # 恢复历史损失（如果需要可视化训练曲线）
            epoch_loss_values = checkpoint.get('epoch_loss_values', [])
            cls_loss_value = checkpoint.get('cls_loss_value', [])
            transfer_loss_value = checkpoint.get('transfer_loss_value', [])
            flow_loss_value = checkpoint.get('flow_loss_value', [])
            reg_loss_values = checkpoint.get('reg_loss_values', [])
            # sum_ssim1 = checkpoint.get('sum_ssim1', 1.0)
            # sum_ssim2 = checkpoint.get('sum_ssim2', 1.0)
        else:
            print(f"Checkpoint file {checkpoint_file} not found. Starting training from scratch.")

    """开始训练循环"""
    for epoch_num in range(start_epoch, num_epoch):
        registration_net.train()  # 设置模型为训练模式

        epoch_total_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_transfer_loss = 0.0
        epoch_flow_loss = 0.0
        epoch_reg_loss = 0.0

        # 调整 parameter，但在这里可能仅作为 regFusion_loss 的一个输入，不影响配准核心
        # parameter 用于调整融合损失中两个SSIM项的权重，在这里可以设置为一个固定值
        # 比如 paper 中的 µ = N∑ n=1 L(n) ssim(Ifuse, IB)/ N∑ n=1 L(n) ssim(Ifuse, ĨA)
        # 在纯配准训练中，可以简化为 0.5 甚至 1.0，或者直接从 regFusion_loss 中移除
        # 为了保持与原代码的regFusion_loss接口一致，保留parameter的传递
        current_parameter = (sum_ssim2) / (sum_ssim1) if sum_ssim1 != 0 else 1.0
        sum_ssim1 = 0.0  # 重新初始化，用于当前 epoch 统计
        sum_ssim2 = 0.0

        epoch_iterator = tqdm(
            train_dataloader, desc=f'Epoch {epoch_num + 1}/{num_epoch} (Loss: N/A)', ncols=150, leave=True, position=0)

        for step, batch in enumerate(epoch_iterator):
            # 将数据移动到设备
            # img1: 参考图像 (MRI), img2: 浮动图像 (PET/CT), flow_GT: 真实形变场 (如果有)
            # label1, label2: 模态分类标签 (用于 MDF-FR 的 Lce1, Lce2)
            # img1_2: 原始代码中是 img1 的 YCbCr 转换，这里根据需要处理

            img1, img1_2, img2, flow_GT, label1, label2 = batch

            # 根据模态处理图像通道 (与原始训练代码保持一致)
            if modal == 'CT':
                img1_proc, img2_proc, img1_2_proc = img1.to(device), img2.to(device), img1_2.to(device)
            else:  # PET
                # 原始代码将 PET 转换为 Y 通道，因此这里也这样做
                img1_proc = img1.to(device)
                img2_proc = img2[:, 0, :, :].unsqueeze(1).to(device)
                img1_2_proc = img1_2[:, 0, :, :].unsqueeze(1).to(device)

            # 其他标签和 GT flow
            flow_GT = flow_GT.to(device)
            label1 = label1.to(device)
            label2 = label2.to(device)

            # --- 前向传播：使用我们集成的配准网络 ---
            # aligned_feature1, aligned_feature2 是配准后的深层特征
            # final_flow 是最终的形变场
            # warped_img2 是 img2 根据 final_flow 扭曲后的图像
            aligned_feature1, aligned_feature2, final_flow, warped_img2, \
            AS_F, BS_F, pre1, pre2, feature_pred1, feature_pred2, all_sub_flows = registration_net(img1_proc, img2_proc)

            """计算损失"""
            # 使用原始的 regFusion_loss，但我们只关注配准相关的损失项
            # 注意：regFusion_loss 的参数 `flows` 期望的是一个包含所有局部流的列表
            # 但 MedicalImageRegistrationNet 的 `reg_net` 内部处理了 flows，只返回 `final_flow`
            # 这里需要调整 regFusion_loss 的调用，或修改 regFusion_loss 仅接受 final_flow
            # 为了简化，我们假设 `regFusion_loss` 内部对 `flows` 的使用是可选的，
            # 或者将其设置为一个空列表，只依赖 `final_flow`。
            # 实际上，`regFusion_loss` 期望 `flows` 是一个包含10个局部流的列表。
            # 我们需要让 MedicalImageRegistrationNet 也返回这个列表，或者修改 RegNet_lite
            # 为了不改动 MedicalImageRegistrationNet 的输出结构，我们暂时将 flows 传入一个空列表或 dummy_flows

            # 重新检查 RegNet_lite 的返回：`return f1, f2, flows, flow, flow_neg, flow_pos`
            # 所以 MedicalImageRegistrationNet 的 `reg_net` 调用处应该是这样捕获的：
            # `aligned_feature1, aligned_feature2, all_sub_flows, final_flow = self.reg_net(feature1_reg_input, feature2_reg_input)`
            # 这样 `all_sub_flows` 就可以传入 `regFusion_loss` 了。
            # 我之前的 MedicalImageRegistrationNet 返回值没有包含 `all_sub_flows`，需要补充。

            # 修正 MedicalImageRegistrationNet 的前向返回值，以获取 all_sub_flows
            # (这个修正需要回到 registration_network.py 文件中完成)
            # 假设 MedicalImageRegistrationNet 的 forward 返回值已调整为：
            # `return aligned_feature1, aligned_feature2, final_flow, warped_img2, all_sub_flows`

            # 如果不想改动 MedicalImageRegistrationNet 的 forward，
            # 那么 regFusion_loss 内部对 flows 的使用需要被关闭或修改。
            # 为了保持与原始训练代码的兼容性，我们将回到 `registration_network.py` 补充返回值。

            # 假设 `all_sub_flows` 已经从 `MedicalImageRegistrationNet` 的返回中获取
            # 如果没有，请在 `registration_network.py` 中修改 `MedicalImageRegistrationNet` 的 `forward` 返回值
            # 假设这里获取 `all_sub_flows`
            # dummy_flows = [torch.zeros_like(final_flow)] * 10 # 临时替代，如果 MedicalImageRegistrationNet 不返回

            # Note: regFusion_loss 内部的 `parameter` 是用于融合损失的 µ
            # 在纯配准训练中，可以将其固定或忽略。
            # 为了避免修改 regFusion_loss，我们继续传入它。

            # 确保 img1 是原始的，未经过 project 的
            # 确保 img1_2 是原始的，未经过 project 的

            cls_loss, transfer_loss, flow_loss, fusion_loss, reg_loss, ssim1, ssim2 = regFusion_loss(
                label1, label2,  # For Lce1, Lce2
                pre1, pre2,  # pre1, pre2 from ModalityDiscrepancyRemoval
                feature_pred1, feature_pred2,
                # feature_pred1, feature_pred2 from ModalityDiscrepancyRemoval (classifier_head after Transfer)
                final_flow,  # Your final integrated flow
                # all_sub_flows, # This is needed by regFusion_loss. If MedicalImageRegistrationNet doesn't return it, you need to modify.
                # For now, let's assume `registration_net` returns `all_sub_flows`
                registration_net.reg_net.all_flows_fwd + registration_net.reg_net.all_flows_bwd,  # 临时方法获取，但不太优雅
                warped_img2, flow_GT,  # For Lconsis
                img1_proc, img1_2_proc, fusion_img=None,
                # For Lstruct, Linten, Lgrad (fusion_img here is None as we don't train fusion_net)
                parameter=current_parameter  # For fusion loss weighting
            )
            # 在纯配准训练中，fusion_loss 应该被忽略或设置为0
            # 原始代码中 fusion_img 传入 fusion_net 的输出，我们这里没有 fusion_net
            # 所以需要确保 regFusion_loss 能够在 fusion_img=None 时正常工作，或者我们直接将其设置为0

            # 过滤掉与融合相关的损失
            loss = cls_loss + transfer_loss + flow_loss + reg_loss  # 移除 fusion_loss

            epoch_total_loss += loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_transfer_loss += transfer_loss.item()
            epoch_flow_loss += flow_loss.item()
            epoch_reg_loss += reg_loss.item()

            # 原始代码中 ssim1, ssim2 用于计算 parameter，可以保留
            sum_ssim1 += ssim1.item()
            sum_ssim2 += ssim2.item()

            # 优化器清零梯度
            optimizer.zero_grad()

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # 更新 tqdm 进度条
            epoch_iterator.set_description(
                f"Epoch {epoch_num + 1}/{num_epoch} (Loss: {loss.item():.4f}, Cls: {cls_loss.item():.4f}, Transfer: {transfer_loss.item():.4f}, Flow: {flow_loss.item():.4f}, Reg: {reg_loss.item():.4f})"
            )

        # 每个 epoch 结束时更新学习率
        scheduler.step()

        # 原始代码中的 transfer.modal_dis.load_state_dict(transfer.classifier.state_dict())
        # 这段逻辑看起来是将 ModelTransfer_lite 内部的 modal_dis 的参数更新为 classifier 的参数。
        # 在我们的 MedicalImageRegistrationNet 中，ModalityDiscrepancyRemoval 扮演了 classifier 的角色，
        # 并且我们只实例化了 ModalityDiscrepancyRemoval，没有 modal_dis。
        # 如果 ModalityDiscrepancyRemoval 的参数是在训练过程中直接更新的，
        # 则不需要额外的 `load_state_dict` 操作。
        # 这里需要检查论文中 MDF-FR 的具体更新策略。
        # 鉴于我们只保留了一个 `ModalityDiscrepancyRemoval` 实例，并且它的参数是可训练的，
        # 这一行可能不再需要。或者，如果 `modality_feature_embedding1` 和 `modality_feature_embedding2`
        # 应该共享参数或者互相更新，则需要特别处理。
        # 原始代码中是 `transfer.modal_dis.load_state_dict(transfer.classifier.state_dict())`
        # 这暗示 `classifier` 和 `modal_dis` 是两个独立的模块，但 `modal_dis` 的权重会被 `classifier` 的权重覆盖。
        # 在你的 `MedicalImageRegistrationNet` 中，`modality_feature_embedding1` 和 `modality_feature_embedding2`
        # 是两个独立的 `ModalityDiscrepancyRemoval` 实例。如果它们应该共享权重，需要修改 `__init__`。
        # 暂时我们假设它们是独立的，并且通过损失函数各自学习。

        # 记录每个 epoch 的平均损失
        epoch_loss_values.append(epoch_total_loss / (step + 1))
        cls_loss_value.append(epoch_cls_loss / (step + 1))
        transfer_loss_value.append(epoch_transfer_loss / (step + 1))
        flow_loss_value.append(epoch_flow_loss / (step + 1))
        reg_loss_values.append(epoch_reg_loss / (step + 1))

        print(f'{Fore.RED}Epoch {epoch_num + 1} Avg Total Loss: {epoch_loss_values[-1]:.4f}{Style.RESET_ALL}')

        """保存检查点"""
        # 每 5 个 epoch 保存一次完整的检查点
        if (epoch_num + 1) % 5 == 0:
            checkpoint = {
                "epoch": epoch_num,
                "registration_net_state_dict": registration_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),  # 保存学习率调度器状态
                "epoch_loss_values": epoch_loss_values,
                "cls_loss_value": cls_loss_value,
                "transfer_loss_value": transfer_loss_value,
                "flow_loss_value": flow_loss_value,
                'reg_loss_values': reg_loss_values,
                # "sum_ssim1": sum_ssim1, # 如果需要恢复这些统计量
                # "sum_ssim2": sum_ssim2,
            }
            # 保存到 epoch_checkpoints 子文件夹
            path_checkpoint = os.path.join(specific_checkpoint_path,
                                           f'registration_checkpoint_epoch_{epoch_num + 1:04d}.pth')
            torch.save(checkpoint, path_checkpoint)
            print(f"Saved checkpoint to {path_checkpoint}")

        # 每 epoch 结束时保存最新的检查点，方便恢复训练
        latest_checkpoint_path = os.path.join(specific_checkpoint_path, 'registration_checkpoint_latest.pth')
        torch.save(checkpoint, latest_checkpoint_path)

    # 训练结束后保存最终模型权重
    final_model_path = os.path.join(checkpoint_dir, f'MedicalImageRegistrationNet_final_{modal}.pth')
    torch.save(registration_net.state_dict(), final_model_path)
    print("Training is completed and final model saved.")


if __name__ == '__main__':
    modal = 'CT'  # 或 'PET'
    train_batch_size = 8  # 根据你的GPU显存调整
    lr = 5e-5
    num_epoch = 3000
    beta1 = 0.9
    beta2 = 0.999
    resume = False  # 设置为 True 可以从上次保存的检查点恢复训练

    train(modal,
          train_batch_size,
          lr,
          num_epoch,
          beta1,
          beta2,
          resume)