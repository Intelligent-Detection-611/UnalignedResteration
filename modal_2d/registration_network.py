import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# 从 Restormer.py 导入 Restormer 模型
# 确保 Restormer.py 在同级目录或其导入路径可达
from modal_2d.Restormer import Restormer

# 从 utils_2d.warp.py 导入 Warper2d 和 warp2D
# 确保 utils_2d/warp.py 在其导入路径可达
from utils_2d.warp import Warper2d, warp2D

# 从 classifier.py 导入 VitBlock 和 PatchEmbedding2D
# 确保 classifier.py 在同级目录或其导入路径可达
from modal_2d.classifier import VitBlock, PatchEmbedding2D

# 从 RegFusion_lite.py 导入 Classifier_lite 和 model_classifer_lite
# 因为它们是在原始 RegFusion_lite.py 中定义的
from modal_2d.RegFusion_lite import Classifier_lite, model_classifer_lite

# --- 辅助函数和类 (从 RegFusion_lite.py 复制过来) ---

# 注：这里的 image_warp_func (warp2D 的实例) 在此独立配准网络中可能未直接使用
# 但为了保持完整性，保留其定义
image_warp_func = warp2D()


def project(x, image_size):
    """将 Transformer 输出的特征从 (b, num_patches, hidden_dim) 转换回 (b, hidden_dim, H/p, W/p)"""
    W, H = image_size[0], image_size[1]
    # 假设 patch_size 是 16，所以 num_patches = (W/16) * (H/16)
    # 这里的 w, h 对应于特征图的尺寸
    w_feat = W // 16
    h_feat = H // 16
    x = rearrange(x, 'b (w h) hidden -> b hidden w h', w=w_feat, h=h_feat)
    return x


# -----------------------------------------------------------------------------
# 关键修改点：img_warp 函数的参数顺序 (已根据 warp.py 中 Warper2d 的实际定义修正)
def img_warp(flow_field, image_to_warp):
    """使用 Warper2d 进行图像/特征扭曲"""
    # 根据 utils_2d/warp.py 中 Warper2d 的 forward(self, flow, img) 定义，
    # 第一个参数是 flow，第二个参数是 img (待扭曲的图像/特征)
    return Warper2d()(flow_field, image_to_warp)


# -----------------------------------------------------------------------------

def _integrate_flows(flow1, flow2, flow3, flow4, flow5, flow6, flow7, flow8, flow9, flow10):
    """
    整合多步预测的形变场，将所有局部流上采样到原始图像尺寸并累加。
    此函数模拟原始 RegFusion_lite.py 中的 flow_integration_ir 逻辑。
    """
    # 定义上采样模块
    up1 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
    up2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
    up3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
    up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    # 执行上采样和缩放
    flow1_up, flow2_up = up1(flow1) * 16, up1(flow2) * 16
    flow3_up, flow4_up = up2(flow3) * 8, up2(flow4) * 8
    flow5_up, flow6_up = up3(flow5) * 4, up3(flow6) * 4
    flow7_up, flow8_up = up4(flow7) * 2, up4(flow8) * 2

    # flow9, flow10 已经是最终分辨率，不需要上采样和乘因子 (或乘以1)
    flow9_up, flow10_up = flow9 * 1, flow10 * 1  # 明确乘以1

    flow_neg_accum = flow1_up + flow3_up + flow5_up + flow7_up + flow9_up
    flow_pos_accum = flow2_up + flow4_up + flow6_up + flow8_up + flow10_up

    final_integrated_flow = flow_pos_accum - flow_neg_accum  # 最终的整合形变场
    return final_integrated_flow, flow_neg_accum, flow_pos_accum


# --- RegNet_lite 依赖的内部模块 (从 RegFusion_lite.py 复制过来) ---

class CrossAttention(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CrossAttention, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )

    def forward(self, f1, f2):
        f1_hat = f1
        f1 = self.conv1(f1)
        f2 = self.conv2(f2)
        att_map = f1 * f2
        att_shape = att_map.shape
        att_map = torch.reshape(att_map, [att_shape[0], att_shape[1], -1])
        att_map = F.softmax(att_map, dim=2)
        att_map = torch.reshape(att_map, att_shape)
        f1 = f1 * att_map
        f1 = f1 + f1_hat
        return f1


class ResBlk(nn.Module):
    def __init__(self, in_channel):
        super(ResBlk, self).__init__()
        self.feature_output = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

    def forward(self, x):
        return x + self.feature_output(x)


class FusionRegBlk_lite(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FusionRegBlk_lite, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(kernel_size=1, stride=1, padding=0, in_channels=in_channel * 2, out_channels=in_channel),
            nn.LeakyReLU())

        self.crossAtt1 = CrossAttention(in_channel, out_channel)
        self.feature_output = nn.Sequential(
            ResBlk(in_channel),
        )

        self.flow_output = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=2, kernel_size=3, padding=1, stride=1),
            nn.Tanh(),  # Tanh 激活函数，限制输出在 -1 到 1 之间
        )
        # 这个 up1 是用于将当前层的特征上采样，然后作为下一层的 f1_next_feat 输入
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, padding=0,
                                           stride=1),
                                 nn.LeakyReLU(), )

    def forward(self, f_in_from_prev_block, f_cat_from_current_level):
        # f_in_from_prev_block 是来自上一层 FusionRegBlk_lite 的特征（例如 f1_next_feat）
        # f_cat_from_current_level 是当前尺度下拼接的 (f1_current, f2_current)

        f_cat_processed = self.conv1x1(f_cat_from_current_level)
        # 交叉注意力使用 f_in_from_prev_block 和 f_cat_processed
        f_attn = self.crossAtt1(f_in_from_prev_block, f_cat_processed) + self.crossAtt1(f_cat_processed,
                                                                                        f_in_from_prev_block)
        f_res = self.feature_output(f_attn)  # 残差块处理

        flow = self.flow_output(f_res)  # 从特征 f_res 预测形变场 flow
        f_up_for_next_block = self.up1(f_res)  # 用于下一阶段的特征（尺寸会放大）

        return f_up_for_next_block, flow  # 返回上采样后的特征和当前层预测的 flow


class UpBlk(nn.Module):
    def __init__(self, in_c, out_c):
        super(UpBlk, self).__init__()
        self.up_sample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, stride=1),
        )
        self.conv1 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, padding=1, stride=1)
        self.in1 = nn.InstanceNorm2d(num_features=out_c)

    def forward(self, x):
        x = self.up_sample(x)
        x = self.conv1(x)
        x = self.in1(x)
        return F.leaky_relu(x)


class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.channels = channels
        self.up1 = UpBlk(self.channels[0], self.channels[1])  # 256 -> 64
        self.up2 = UpBlk(self.channels[1], self.channels[2])  # 64 -> 32
        self.up3 = UpBlk(self.channels[2], self.channels[3])  # 32 -> 16
        self.up4 = UpBlk(self.channels[3], self.channels[4])  # 16 -> 8


# --- 从 RegFusion_lite.py 复制的 Encoder 和 ModelTransfer_lite 中的核心部分 ---

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.rb1 = Restormer(1, 8)  # 输入通道1，输出8
        self.rb2 = Restormer(8, 3)  # 输入通道8，输出3 (用于 Transformer)

    def forward(self, img):
        # img: (B, 1, H, W)
        f = self.rb1(img)  # (B, 8, H, W)
        f_ = self.rb2(f)  # (B, 3, H, W)
        return f, f_  # f是浅层特征，f_是深层特征，用于输入Transformer


class Transfer(nn.Module):
    # 这是 ModelTransfer_lite 的核心部分
    def __init__(self, num_vit, num_heads):
        super(Transfer, self).__init__()
        self.num_vit = num_vit
        self.num_heads = num_heads
        self.hidden_dim = 256  # 修改为 256 以匹配 RegNet_lite 输入的特征维度
        self.cls1 = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.cls2 = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.VitBLK1 = nn.Sequential()
        for i in range(self.num_vit):
            self.VitBLK1.add_module(name=f'vit{i}',
                                    module=VitBlock(hidden_size=self.hidden_dim,
                                                    num_heads=self.num_heads,
                                                    vit_drop=0.0,
                                                    qkv_bias=False,
                                                    mlp_dim=self.hidden_dim,  # 修改为 256
                                                    mlp_drop=0.0))
        self.VitBLK2 = nn.Sequential()
        for i in range(self.num_vit):
            self.VitBLK2.add_module(name=f'vit{i}',
                                    module=VitBlock(hidden_size=self.hidden_dim,
                                                    num_heads=self.num_heads,
                                                    vit_drop=0.0,
                                                    qkv_bias=False,
                                                    mlp_dim=self.hidden_dim,  # 修改为 256
                                                    mlp_drop=0.0))

    def forward(self, x1, x2, cls1, cls2):
        # x1, x2: (B, num_patches, hidden_dim) 是经过 PatchEmbedding 和 Transformer 后的特征
        # cls1, cls2: (B, hidden_dim) 是分类 token

        # 扩展 cls token 到所有 patch，并执行特征注入
        cls1_expanded = cls1.unsqueeze(dim=1).expand(-1, x1.shape[1], -1)
        cls2_expanded = cls2.unsqueeze(dim=1).expand(-1, x1.shape[1], -1)

        x1_injected = x1 + cls2_expanded  # x1 (from img1) 注入 x2 的模态信息
        x2_injected = x2 + cls1_expanded  # x2 (from img2) 注入 x1 的模态信息

        # 将 class token 拼接回特征序列，用于 VITBLK 处理
        class_token1_vit = self.cls1.expand(x1.shape[0], -1, -1)
        class_token2_vit = self.cls2.expand(x1.shape[0], -1, -1)

        x1_processed = torch.cat((class_token1_vit, x1_injected), dim=1)
        x2_processed = torch.cat((class_token2_vit, x2_injected), dim=1)

        x1_out = self.VitBLK1(x1_processed)
        x2_out = self.VitBLK2(x2_processed)

        # 分离 class token 和 patch features
        new_cls1 = x1_out[:, 0, :]
        new_cls2 = x2_out[:, 0, :]

        # 返回处理后的特征（不包含 class token）和新的 class token
        return x1_out[:, 1:, :], x2_out[:, 1:, :], new_cls1, new_cls2


class ModalityDiscrepancyRemoval(nn.Module):
    # 这是 ModelTransfer_lite 中用于处理模态差异的部分 (MFRH)
    def __init__(self, in_channels, num_heads, num_vit_blk, img_size, patch_size):
        super(ModalityDiscrepancyRemoval, self).__init__()
        # Classifier_lite 包含了 PatchEmbedding 和 VIT Blocks，用于处理特征序列并输出分类token
        self.classifier_head = Classifier_lite(in_c=in_channels, num_heads=num_heads, num_vit_blk=num_vit_blk,
                                               img_size=img_size, patch_size=patch_size)

    def forward(self, img_features):
        # img_features 是 Encoder 的输出 (B, C, H, W)
        # Classifier_lite 的 forward 会先进行 PatchEmbedding，然后通过 VIT Blocks
        # 返回 (predict, cls_token, patch_tokens)
        predict, cls_token, patch_tokens = self.classifier_head(img_features)
        return predict, cls_token, patch_tokens


class RegNet_lite(nn.Module):
    def __init__(self):
        super(RegNet_lite, self).__init__()
        self.channels = [256, 64, 32, 16, 8, 1]

        # 前向配准层 (FRL)
        self.f1_FR1 = FusionRegBlk_lite(in_channel=self.channels[0], out_channel=self.channels[1])
        self.f1_FR2 = FusionRegBlk_lite(in_channel=self.channels[1], out_channel=self.channels[2])
        self.f1_FR3 = FusionRegBlk_lite(in_channel=self.channels[2], out_channel=self.channels[3])
        self.f1_FR4 = FusionRegBlk_lite(in_channel=self.channels[3], out_channel=self.channels[4])
        self.f1_FR5 = FusionRegBlk_lite(in_channel=self.channels[4], out_channel=self.channels[5])

        # 反向配准层 (RRL)
        self.f2_FR1 = FusionRegBlk_lite(in_channel=self.channels[0], out_channel=self.channels[1])
        self.f2_FR2 = FusionRegBlk_lite(in_channel=self.channels[1], out_channel=self.channels[2])
        self.f2_FR3 = FusionRegBlk_lite(in_channel=self.channels[2], out_channel=self.channels[3])
        self.f2_FR4 = FusionRegBlk_lite(in_channel=self.channels[3], out_channel=self.channels[4])
        self.f2_FR5 = FusionRegBlk_lite(in_channel=self.channels[4], out_channel=self.channels[5])

        self.decoder_upsample = Decoder(self.channels)  # 用于对齐后的特征上采样

    def forward(self, f1_init, f2_init):
        # f1_init, f2_init 是来自 ModelTransfer_lite 处理后的特征，维度是 (B, C, H_feat, W_feat)
        # 例如：(B, 256, 16, 16) 如果原始图像是 256x256，patch_size=16 (256/16=16)

        # 保存所有预测的局部形变场
        all_flows_fwd = []  # 正向流
        all_flows_bwd = []  # 反向流

        # 初始化当前层的特征
        f1_current = f1_init
        f2_current = f2_init

        # 初始化 FusionRegBlk_lite 的第一个输入特征（对应论文中的 D_A, D_B）
        # 初始 D_A = F1_A, D_B = F1_B
        f1_next_feat_block_input = f1_init
        f2_next_feat_block_input = f2_init

        # ----------------- 逐步配准迭代 -----------------
        # 迭代 1 (最高特征分辨率，最低语义级别)
        f_cat = torch.cat((f1_current, f2_current), dim=1)  # (B, 512, H_feat, W_feat)
        # f_up_for_next_block是FusionRegBlk_lite的f1_up, flow是FusionRegBlk_lite的flow
        f1_next_feat_block_input, flow1 = self.f1_FR1(f1_next_feat_block_input, f_cat)
        f2_next_feat_block_input, flow2 = self.f2_FR1(f2_next_feat_block_input, f_cat)

        all_flows_fwd.append(flow1)
        all_flows_bwd.append(flow2)

        # 应用当前层预测的 flow 来扭曲特征，用于下一层的输入
        f1_warped_for_next = img_warp(flow1, f1_current)
        f2_warped_for_next = img_warp(flow2, f2_current)

        # 解码器上采样，准备下一层输入维度
        f1_current = self.decoder_upsample.up1(f1_warped_for_next)
        f2_current = self.decoder_upsample.up1(f2_warped_for_next)

        # 迭代 2
        f_cat = torch.cat((f1_current, f2_current), dim=1)
        f1_next_feat_block_input, flow3 = self.f1_FR2(f1_next_feat_block_input, f_cat)
        f2_next_feat_block_input, flow4 = self.f2_FR2(f2_next_feat_block_input, f_cat)

        all_flows_fwd.append(flow3)
        all_flows_bwd.append(flow4)

        f1_warped_for_next = img_warp(flow3, f1_current)
        f2_warped_for_next = img_warp(flow4, f2_current)
        f1_current = self.decoder_upsample.up2(f1_warped_for_next)
        f2_current = self.decoder_upsample.up2(f2_warped_for_next)

        # 迭代 3
        f_cat = torch.cat((f1_current, f2_current), dim=1)
        f1_next_feat_block_input, flow5 = self.f1_FR3(f1_next_feat_block_input, f_cat)
        f2_next_feat_block_input, flow6 = self.f2_FR3(f2_next_feat_block_input, f_cat)

        all_flows_fwd.append(flow5)
        all_flows_bwd.append(flow6)

        f1_warped_for_next = img_warp(flow5, f1_current)
        f2_warped_for_next = img_warp(flow6, f2_current)
        f1_current = self.decoder_upsample.up3(f1_warped_for_next)
        f2_current = self.decoder_upsample.up3(f2_warped_for_next)

        # 迭代 4
        f_cat = torch.cat((f1_current, f2_current), dim=1)
        f1_next_feat_block_input, flow7 = self.f1_FR4(f1_next_feat_block_input, f_cat)
        f2_next_feat_block_input, flow8 = self.f2_FR4(f2_next_feat_block_input, f_cat)

        all_flows_fwd.append(flow7)
        all_flows_bwd.append(flow8)

        f1_warped_for_next = img_warp(flow7, f1_current)
        f2_warped_for_next = img_warp(flow8, f2_current)
        f1_current = self.decoder_upsample.up4(f1_warped_for_next)
        f2_current = self.decoder_upsample.up4(f2_warped_for_next)

        # 迭代 5 (最终尺度)
        f_cat = torch.cat((f1_current, f2_current), dim=1)
        # f1_final_feature_block_output 和 f2_final_feature_block_output 是 FusionRegBlk_lite 最后一层输出的特征
        # 它们被上采样到 in_channel=1 的特征，但实际在论文中 RegNet 的输出是形变场和扭曲后的特征
        # 这里我们只取流
        _, flow9 = self.f1_FR5(f1_next_feat_block_input, f_cat)
        _, flow10 = self.f2_FR5(f2_next_feat_block_input, f_cat)

        all_flows_fwd.append(flow9)
        all_flows_bwd.append(flow10)

        # 最后一次扭曲特征 (这里的 f1_current 和 f2_current 已经是最高分辨率特征)
        f1_warped_final_aligned_feature = img_warp(flow9, f1_current)
        f2_warped_final_aligned_feature = img_warp(flow10, f2_current)

        # 组合所有局部形变场以得到最终的形变场
        final_flow, final_flow_neg, final_flow_pos = _integrate_flows(
            all_flows_fwd[0], all_flows_bwd[0],  # flow1, flow2
            all_flows_fwd[1], all_flows_bwd[1],  # flow3, flow4
            all_flows_fwd[2], all_flows_bwd[2],  # flow5, flow6
            all_flows_fwd[3], all_flows_bwd[3],  # flow7, flow8
            all_flows_fwd[4], all_flows_bwd[4]  # flow9, flow10
        )

        # 返回最终对齐的特征和形变场
        # 返回值顺序调整，以匹配 registration_train.py 中的捕获
        return f1_warped_final_aligned_feature, f2_warped_final_aligned_feature, final_flow, \
               all_flows_fwd + all_flows_bwd  # 返回所有局部流的列表


# --- 整合为一个独立的配准网络 ---
class MedicalImageRegistrationNet(nn.Module):
    def __init__(self, img_size=(256, 256), patch_size=16, num_vit_blk=2, num_heads=4):
        super(MedicalImageRegistrationNet, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        # 1. 特征提取器 (Encoder)
        self.encoder = Encoder()

        # 2. 模态差异消除和特征转换器 (MDF-FR 的一部分)
        # 这里的 in_channels=3 是 Encoder 的 rb2 输出通道
        # modality_feature_embedding1 和 2 用于 Lce1 (原始模态鉴别)
        self.modality_feature_embedding1 = ModalityDiscrepancyRemoval(in_channels=3, num_heads=num_heads,
                                                                      num_vit_blk=num_vit_blk, img_size=img_size,
                                                                      patch_size=patch_size)
        self.modality_feature_embedding2 = ModalityDiscrepancyRemoval(in_channels=3, num_heads=num_heads,
                                                                      num_vit_blk=num_vit_blk, img_size=img_size,
                                                                      patch_size=patch_size)

        # Transfer 模块 (用于特征注入和转换)
        self.transfer_block = Transfer(num_vit=num_vit_blk, num_heads=num_heads)

        # 用于 Lce2 损失的模态鉴别器，处理转换后的特征。
        # 它的输入通道应是 feature_transformed 的 hidden_dim (256)
        # 这是为了模拟原始代码中的 `transfer.modal_dis`
        self.modal_dis_head = model_classifer_lite(in_c=256, num_heads=num_heads, num_vit_blk=num_vit_blk,
                                                   img_size=img_size, patch_size=patch_size)

        # 3. 配准网络本身 (RegNet_lite)
        self.reg_net = RegNet_lite()

    def forward(self, img1, img2):
        """
        img1: 参考图像 (e.g., MRI), (B, 1, H, W)
        img2: 浮动图像 (e.g., PET), (B, 1, H, W)

        返回：
        aligned_feature1: 配准后的特征1 (来自 img1)，(B, 8, H, W)
        aligned_feature2: 配准后的特征2 (来自 img2)，(B, 8, H, W)
        final_flow: 最终整合的形变场，(B, 2, H, W)
        warped_img2: 根据 final_flow 扭曲后的 img2 图像，(B, 1, H, W)
        AS_F: img1 的浅层特征，(B, 8, H, W)
        BS_F: img2 的浅层特征，(B, 8, H, W)
        pre1: img1 原始特征的分类预测 (Lce1)，(B, 2)
        pre2: img2 原始特征的分类预测 (Lce1)，(B, 2)
        feature_pred1: img1 转换后特征的分类预测 (Lce2)，(B, 2)
        feature_pred2: img2 转换后特征的分类预测 (Lce2)，(B, 2)
        all_local_flows: RegNet_lite 内部生成的所有局部流的列表 (10个流)
        """
        # 步骤 1: 特征提取 (Encoder)
        # AS_F 和 BS_F 是浅层特征 (B, 8, H, W)，它们在原始代码中会给 FusionNet 使用
        AS_F, feature1_enc = self.encoder(img1)  # feature1_enc 是深层特征 (B, 3, H, W)
        BS_F, feature2_enc = self.encoder(img2)  # feature2_enc 是深层特征 (B, 3, H, W)

        # 步骤 2: 模态差异消除和特征转换 (MDF-FR)
        # pre1, pre2 用于 Lce1 损失，它们是原始模态特征的分类结果
        # x1_patch_tokens, x2_patch_tokens 是经过 PatchEmbedding 和 VIT Blocks 后的序列特征
        # cls1, cls2 是对应的分类 token
        pre1, cls1, x1_patch_tokens = self.modality_feature_embedding1(feature1_enc)  # (B, 2), (B, 256), (B, N, 256)
        pre2, cls2, x2_patch_tokens = self.modality_feature_embedding2(feature2_enc)  # (B, 2), (B, 256), (B, N, 256)

        # 经过 Transfer Block 进行特征注入和转换
        # feature1_transformed, feature2_transformed 是转换后的序列特征
        # new_cls1, new_cls2 是转换后的分类 token (可能在后续不直接使用)
        feature1_transformed, feature2_transformed, new_cls1, new_cls2 = self.transfer_block(
            x1_patch_tokens, x2_patch_tokens, cls1, cls2
        )

        # 使用 modal_dis_head (model_classifer_lite) 对转换后的特征进行分类，用于 Lce2 损失
        # 目标是希望转换后的特征不再具有模态特异性，因此分类器应该表现不好（或预测为“融合”类别）
        feature_pred1, _, _ = self.modal_dis_head(feature1_transformed)  # (B, 2)
        feature_pred2, _, _ = self.modal_dis_head(feature2_transformed)  # (B, 2)

        # 将 Transformer 输出的特征（patch tokens）转换回图像特征图格式 for RegNet_lite
        # project 函数将 (B, N_patches, hidden_dim) 转换为 (B, hidden_dim, H/16, W/16)
        # hidden_dim 为 256
        feature1_reg_input = project(feature1_transformed, self.img_size).contiguous()  # (B, 256, H/16, W/16)
        feature2_reg_input = project(feature2_transformed, self.img_size).contiguous()  # (B, 256, H/16, W/16)

        # 步骤 3: 配准 (RegNet_lite)
        # reg_net 会返回最终对齐的特征、所有局部流列表、以及最终整合的形变场
        aligned_feature1, aligned_feature2, all_local_flows, final_flow = self.reg_net(feature1_reg_input,
                                                                                       feature2_reg_input)

        # 步骤 4: 生成配准后的图像 (使用最终的 flow 对原始图像进行扭曲)
        # final_flow 是由 _integrate_flows 生成的，其尺寸是原始图像尺寸 (H, W)
        # img2 是原始输入图像 (B, 1, H, W)
        warped_img2 = img_warp(final_flow, img2)  # 将 img2 配准到 img1 的空间

        # 返回所有训练所需的信息
        return aligned_feature1, aligned_feature2, final_flow, warped_img2, \
               AS_F, BS_F, pre1, pre2, feature_pred1, feature_pred2, all_local_flows


# --- 测试独立配准网络 (在 main 函数中) ---
if __name__ == '__main__':
    # 确保你已经安装了 PyTorch 和其他依赖
    # 如果遇到导入错误，请检查文件路径是否正确
    # 例如，如果 modal_2d 是一个包，可能需要确保其在 Python 路径中
    # 或者将这些文件直接放在同一目录下

    # 模拟输入图像
    img_size = (256, 256)
    # img1: MRI (单通道)
    img1 = torch.randn(1, 1, img_size[0], img_size[1])  # Batch, Channel, H, W
    # img2: PET (单通道，因为在代码中 PET 会被转为 YCbCr，然后只取 Y 通道)
    # 如果你的 PET 原始输入是 RGB，请确保在传入这里之前将其转换为单通道
    img2 = torch.randn(1, 1, img_size[0], img_size[1])

    # 如果要使用 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    img1 = img1.to(device)
    img2 = img2.to(device)

    # 实例化独立的配准网络
    model = MedicalImageRegistrationNet(img_size=img_size).to(device)
    # 冻结 modal_dis_head 的参数，与原始训练代码保持一致
    for par in model.modal_dis_head.parameters():
        par.requires_grad = False

    model.eval()  # 评估模式

    print("Running forward pass through the registration network...")
    with torch.no_grad():
        aligned_feature1, aligned_feature2, final_flow, warped_img2, \
        AS_F, BS_F, pre1, pre2, feature_pred1, feature_pred2, all_local_flows = model(img1, img2)

    print(f"Input Image 1 Shape: {img1.shape}")
    print(f"Input Image 2 Shape: {img2.shape}")
    print(f"Aligned Feature 1 Shape: {aligned_feature1.shape}")  # (B, 8, H, W)
    print(f"Aligned Feature 2 Shape: {aligned_feature2.shape}")  # (B, 8, H, W)
    print(f"Final Flow Shape: {final_flow.shape}")  # (B, 2, H, W)
    print(f"Warped Image 2 Shape: {warped_img2.shape}")  # (B, 1, H, W)
    print(f"AS_F Shape: {AS_F.shape}")  # (B, 8, H, W)
    print(f"BS_F Shape: {BS_F.shape}")  # (B, 8, H, W)
    print(f"Pre1 Shape: {pre1.shape}")  # (B, 2)
    print(f"Pre2 Shape: {pre2.shape}")  # (B, 2)
    print(f"Feature Pred1 Shape: {feature_pred1.shape}")  # (B, 2)
    print(f"Feature Pred2 Shape: {feature_pred2.shape}")  # (B, 2)
    print(f"Number of local flows: {len(all_local_flows)}")  # 应该有 10 个
    for i, f in enumerate(all_local_flows):
        print(f"  Local Flow {i + 1} Shape: {f.shape}")

    print("Registration network extraction and test complete.")

    # 可选：保存一些结果进行可视化
    # from torchvision.utils import save_image
    # import os
    #
    # if not os.path.exists("./registration_outputs"):
    #     os.makedirs("./registration_outputs")
    # save_image(img1.cpu(), "./registration_outputs/original_mri.png")
    # save_image(img2.cpu(), "./registration_outputs/original_pet.png")
    # save_image(warped_img2.cpu(), "./registration_outputs/warped_pet_aligned.png")
    # 要可视化特征图，可能需要将其归一化到 [0, 1] 范围
    # save_image(aligned_feature1.cpu().mean(dim=1, keepdim=True), "./registration_outputs/aligned_feature1_mean.png")
    # save_image(aligned_feature2.cpu().mean(dim=1, keepdim=True), "./registration_outputs/aligned_feature2_mean.png")