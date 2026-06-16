"""TFBG 光谱求逆 - 逆向模型 (Inverse Model)"""
# ============================================================================
# 第一部分：工具加载
# ============================================================================
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 深度学习框架
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 物理约束所需的网络结构（封装自 正向模型.py）
from 正向模型 import ForwardNet

# 可视化
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 第二部分：数据加载
# ============================================================================

class InverseDataLoader:
    """逆向模型数据加载器"""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.wavelength_range = (1520, 1600)
        self.spectrum_points = 4001

    def parse_filename(self, filename):
        """从文件名解析参数"""
        name = filename.replace('.txt', '')
        params = {}

        match = re.search(r'p=([\d.]+)', name)
        if match:
            params['p'] = float(match.group(1))

        match = re.search(r'光栅角度=([\d.e+-]+)', name)
        if match:
            params['tilt_angle'] = float(match.group(1))

        match = re.search(r'光栅长度=(\d+)', name)
        if match:
            params['long'] = float(match.group(1))

        match = re.search(r'光栅周期=([\d.e+-]+)', name)
        if match:
            params['period'] = float(match.group(1))

        match = re.search(r'调制深度=([\d.e+-]+)', name)
        if match:
            params['delta_n'] = float(match.group(1))

        match = re.search(r'sigma=([\d.e+-]+)', name)
        if match:
            params['sigma'] = float(match.group(1))

        return params

    def load_spectrum(self, filepath):
        """加载单个光谱文件"""
        data = pd.read_csv(filepath, sep='\t', header=None)
        return data.values

    def load_all_data(self):
        """加载所有数据"""
        txt_files = list(self.data_dir.glob('*.txt'))
        print(f"找到 {len(txt_files)} 个光谱文件")

        spectra = []
        params_list = []

        for filepath in sorted(txt_files):
            try:
                spectrum = self.load_spectrum(filepath)
                params = self.parse_filename(filepath.name)

                # 只取前3个需要回归的参数
                param_array = np.array([
                    params['tilt_angle'],
                    params['delta_n'],
                    params['sigma']
                ])

                spectra.append(spectrum[:, 1:3])  # 透射响应1和透射响应2
                params_list.append(param_array)

            except Exception as e:
                print(f"加载失败 {filepath.name}: {e}")

        spectra = np.array(spectra)
        params_list = np.array(params_list)

        print(f"光谱数据形状: {spectra.shape}")
        print(f"参数数据形状: {params_list.shape}")

        return {'spectra': spectra, 'params': params_list}

    def normalize_spectra(self, spectra):
        """归一化光谱"""
        spectra_norm = spectra.copy()

        for i in range(len(spectra)):
            v = spectra[i, :, 0]
            v_min = v.min()
            v_max = v.max()
            v_range = v_max - v_min
            if v_range > 0:
                spectra_norm[i, :, 0] = 2 * (v - v_min) / v_range - 1
            else:
                spectra_norm[i, :, 0] = v - v_min  # 平移到0

        # 透射响应2：min-max 归一化到 [-1, 1]
        for i in range(len(spectra)):
            v = spectra[i, :, 1]
            v_min = v.min()
            v_max = v.max()
            v_range = v_max - v_min
            if v_range > 0:
                spectra_norm[i, :, 1] = 2 * (v - v_min) / v_range - 1
            else:
                spectra_norm[i, :, 1] = v - v_min

        return spectra_norm

    def normalize_params(self, params):
        """归一化参数到 [0, 1]"""
        params_norm = params.copy()
        # tilt_angle: 7.0 - 7.5
        params_norm[:, 0] = (params[:, 0] - 7.0) / 0.5
        # delta_n: 4.0e-4 - 6.0e-4
        params_norm[:, 1] = (params[:, 1] - 4.0e-4) / 2.0e-4
        # sigma: 6.0e-4 - 7.0e-4
        params_norm[:, 2] = (params[:, 2] - 6.0e-4) / 1.0e-4
        return params_norm

    def denormalize_params(self, params_norm):
        """反归一化参数"""
        params = params_norm.copy()
        params[:, 0] = params_norm[:, 0] * 0.5 + 7.0
        params[:, 1] = params_norm[:, 1] * 2.0e-4 + 4.0e-4
        params[:, 2] = params_norm[:, 2] * 1.0e-4 + 6.0e-4
        return params

    def split_data(self, data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        """划分数据集"""
        np.random.seed(seed)
        n = len(data['spectra'])
        indices = np.random.permutation(n)

        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        def split_by_indices(d, indices):
            return {
                'spectra': d['spectra'][indices],
                'params': d['params'][indices]
            }

        return (
            split_by_indices(data, train_idx),
            split_by_indices(data, val_idx),
            split_by_indices(data, test_idx)
        )


class InverseDataset(Dataset):
    """逆向模型数据集"""

    def __init__(self, spectra, params):
        self.spectra = torch.FloatTensor(spectra)
        self.params = torch.FloatTensor(params)

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        return self.spectra[idx], self.params[idx]


# 测试数据加载
if __name__ == "__main__":
    DATA_DIR = Path(__file__).parent / "data"

    loader = InverseDataLoader(DATA_DIR)
    data = loader.load_all_data()

    # 归一化
    spectra_norm = loader.normalize_spectra(data['spectra'])
    params_norm = loader.normalize_params(data['params'])

    print(f"\n归一化后:")
    print(f"光谱范围: [{spectra_norm.min():.3f}, {spectra_norm.max():.3f}]")
    print(f"参数范围: [{params_norm.min():.3f}, {params_norm.max():.3f}]")

    # 划分数据集
    train_data, val_data, test_data = loader.split_data({
        'spectra': spectra_norm,
        'params': params_norm
    }, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    print(f"\n数据集划分:")
    print(f"训练集: {len(train_data['spectra'])} 样本")
    print(f"验证集: {len(val_data['spectra'])} 样本")
    print(f"测试集: {len(test_data['spectra'])} 样本")

    # 参数统计
    print(f"\n参数统计 (原始):")
    print(f"tilt_angle: {data['params'][:, 0].min():.2f} - {data['params'][:, 0].max():.2f}")
    print(f"delta_n: {data['params'][:, 1].min():.2e} - {data['params'][:, 1].max():.2e}")
    print(f"sigma: {data['params'][:, 2].min():.2e} - {data['params'][:, 2].max():.2e}")


# ============================================================================
# 第三部分：神经网络（1dCNN + 全连接 + 残差）
# ============================================================================

class SinActivation(nn.Module):
    """Sin激活函数"""
    def forward(self, x):
        return torch.sin(x)


class ResidualBlock(nn.Module):
    """残差模块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.act = SinActivation()
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class InverseNet(nn.Module):
    """逆向网络：光谱 → 参数

    输入: 光谱 (4001, 2)
    输出: 3个参数 (tilt_angle, delta_n, sigma)
    """

    def __init__(self, input_len=4001, input_channels=2, output_dim=3):
        super().__init__()

        self.cnn = nn.Sequential(
            # 输入: (batch, 2, 4001)
            nn.Conv1d(input_channels, 64, kernel_size=16, stride=4),
            nn.BatchNorm1d(64),
            SinActivation(),
            ResidualBlock(64, 64),

            nn.Conv1d(64, 128, kernel_size=8, stride=4),
            nn.BatchNorm1d(128),
            SinActivation(),
            ResidualBlock(128, 128),

            nn.Conv1d(128, 256, kernel_size=8, stride=4),
            nn.BatchNorm1d(256),
            SinActivation(),
            ResidualBlock(256, 256),

            nn.Conv1d(256, 512, kernel_size=8, stride=4),
            nn.BatchNorm1d(512),
            SinActivation(),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            SinActivation(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            SinActivation(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim),
            # 无激活，让物理约束来限制范围
        )

    def forward(self, x):
        """
        Args:
            x: 光谱 (batch, input_len, input_channels)
        Returns:
            参数 (batch, output_dim)
        """
        batch_size = x.size(0)

        # 变换维度: (batch, len, channels) -> (batch, channels, len)
        x = x.transpose(1, 2)

        x = self.cnn(x)

        x = self.global_pool(x)
        x = x.view(batch_size, -1)

        x = self.fc(x)

        return x


def test_inverse_net():
    """测试逆向网络"""
    print("\n测试 InverseNet...")
    model = InverseNet(input_len=4001, input_channels=2, output_dim=3)
    x = torch.randn(4, 4001, 2)  # 4个样本，光谱
    y = model(x)
    print(f"输入: {x.shape}")
    print(f"输出: {y.shape}")
    assert y.shape == (4, 3), "InverseNet 输出形状错误"
    print("InverseNet 测试通过!")


if __name__ == "__main__":
    test_inverse_net()


# ============================================================================
# 第四部分：物理约束（自包含实现，依赖 逆向模型.py + 正向模型.py 的 ForwardNet）
# ============================================================================
#
# 物理约束完全在本文件内实现，含 4 项子损失：ℒ_param / ℒ_bc / ℒ_pde / ℒ_energy。
# 其中 ℒ_bc/ℒ_pde/ℒ_energy 通过 ForwardNet 反算光谱后施加约束。


class InversePhysicsConstraint:
    """逆向模型物理约束 - 使用正向模型作为微分求解器"""

    def __init__(self, device='cuda', w_pde=1.0, w_bc=1.0, w_energy=0.1, w_param=1.0,
                 forward_model_path=None):
        self.device = torch.device(device)
        self.w_pde = w_pde
        self.w_bc = w_bc
        self.w_energy = w_energy
        self.w_param = w_param

        # 加载正向模型
        self.forward_net = None
        if forward_model_path and Path(forward_model_path).exists():
            self.forward_net = ForwardNet(input_dim=3, output_len=4001, output_channels=2).to(device)
            self.forward_net.load_state_dict(torch.load(forward_model_path, map_location=device))
            self.forward_net.eval()
            print(f"已加载正向模型: {forward_model_path}")
        else:
            raise FileNotFoundError(f"未找到正向模型: {forward_model_path}")

    def compute_loss(self, params_pred):
        """计算物理约束损失

        Args:
            params_pred: 预测的参数 (batch, 3) - [tilt_angle, delta_n, sigma]

        Returns:
            total_loss: 总损失
            loss_dict: 各分项损失
        """
        total_loss = 0.0
        loss_dict = {'pde': 0.0, 'bc_s': 0.0, 'bc_p': 0.0, 'energy': 0.0, 'param': 0.0}

        # ℒ_param: 参数范围约束 (0, 1)
        below = torch.relu(-params_pred)
        above = torch.relu(params_pred - 1.0)
        param_violation = below + above
        loss_dict['param'] = param_violation.mean().item()
        if self.w_param > 0:
            total_loss += self.w_param * param_violation.mean()

        # 2. ℒ_bc_s + ℒ_bc_p + ℒ_pde: 使用正向模型检验
        # 关键修复：去掉 torch.no_grad() 以允许梯度反向传播
        # 正向模型已设置为 eval() 模式且权重冻结，仅作为微分求解器
        if self.forward_net is not None:
            # 不使用 torch.no_grad()，允许梯度流通过用于反向传播
            spectrum_pred = self.forward_net(params_pred)  # (batch, 4001, 2)

            # 提取两个偏振方向的透射谱
            if spectrum_pred.dim() == 3:
                S = spectrum_pred[:, :, 0]  # 透射响应1
                P = spectrum_pred[:, :, 1]  # 透射响应2
            else:
                S = spectrum_pred[..., 0]
                P = spectrum_pred[..., 1]

            # ℒ_bc_s: 透射响应1 范围约束 (应约在 -30dB ~ 0dB)
            bc_s_below = torch.relu(-30.0 - S)
            bc_s_above = torch.relu(S)
            bc_s_violation = bc_s_below + bc_s_above
            loss_dict['bc_s'] = bc_s_violation.mean().item()
            if self.w_bc > 0:
                total_loss += self.w_bc * bc_s_violation.mean()

            # ℒ_bc_p: 透射响应2 范围约束 (应约在 -30dB ~ 0dB)
            bc_p_below = torch.relu(-30.0 - P)
            bc_p_above = torch.relu(P)
            bc_p_violation = bc_p_below + bc_p_above
            loss_dict['bc_p'] = bc_p_violation.mean().item()
            if self.w_bc > 0:
                total_loss += self.w_bc * bc_p_violation.mean()

            # ℒ_pde: 光谱平滑性约束 (S 和 P 两个通道相邻点差值都应 ≤ 0.1 dB)
            S_diff = S[:, 1:] - S[:, :-1]
            P_diff = P[:, 1:] - P[:, :-1]
            S_violation = torch.relu(torch.abs(S_diff) - 0.1)
            P_violation = torch.relu(torch.abs(P_diff) - 0.1)
            # 两通道合并为单个 pde 子项
            pde_loss = (S_violation.mean() + P_violation.mean()) * 0.5
            loss_dict['pde'] = pde_loss.item()
            if self.w_pde > 0:
                total_loss += self.w_pde * pde_loss

            # ℒenergy: 能量守恒约束 (平均功率应约1，即0dB)
            T = torch.pow(10.0, S / 10.0)  # dB -> 线性
            T_mean = torch.mean(T)
            energy_violation = torch.relu(T_mean - 1.1) + torch.relu(0.9 - T_mean)
            loss_dict['energy'] = energy_violation.item()
            if self.w_energy > 0:
                total_loss += self.w_energy * energy_violation

        loss_dict['total'] = float(total_loss) if not isinstance(total_loss, torch.Tensor) else total_loss.item()
        return total_loss, loss_dict


# ============================================================================
# 第五部分：保存和输出
# ============================================================================

class ModelExporter:
    """模型导出和保存"""

    def __init__(self, save_dir='./output'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

    def save_model(self, model, filename):
        save_path = self.save_dir / filename
        torch.save(model.state_dict(), save_path)
        print(f"模型已保存: {save_path}")
        return save_path


# ============================================================================
# 主训练入口
# ============================================================================

def main_training():
    print("=" * 60)
    print("TFBG 光谱求逆 - 逆向模型训练")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    DATA_DIR = Path(__file__).parent / "data"
    SAVE_DIR = Path(__file__).parent / "code-inverse" / "output"
    SAVE_DIR_PATH = Path(SAVE_DIR)
    SAVE_DIR_PATH.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    print("\n加载数据...")
    loader = InverseDataLoader(DATA_DIR)
    data = loader.load_all_data()

    # 归一化
    spectra_norm = loader.normalize_spectra(data['spectra'])
    params_norm = loader.normalize_params(data['params'])

    # 划分数据集
    train_data, val_data, test_data = loader.split_data({
        'spectra': spectra_norm,
        'params': params_norm
    }, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    print(f"训练集: {len(train_data['spectra'])} 样本")
    print(f"验证集: {len(val_data['spectra'])} 样本")
    print(f"测试集: {len(test_data['spectra'])} 样本")

    # 2. 创建DataLoader
    train_dataset = InverseDataset(train_data['spectra'], train_data['params'])
    val_dataset = InverseDataset(val_data['spectra'], val_data['params'])
    test_dataset = InverseDataset(test_data['spectra'], test_data['params'])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 3. 初始化模型
    print("\n初始化模型...")
    model = InverseNet(input_len=4001, input_channels=2, output_dim=3).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 4. 训练设置
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    # 余弦退火：T_max=总epoch上限，eta_min=最小学习率
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=600, eta_min=1e-6)

    # 5. 训练循环
    print("\n开始训练...")
    best_val_loss = float('inf')
    patience = 50  # 早停耐心参数（数据损失和物理损失连续50轮无改善则停止）
    min_epochs = 100  # 最小训练轮数（避免早停误触发）
    patience_counter = 0
    train_losses = []
    val_losses = []
    phy_losses = []

    # 联合收敛判断：数据损失阈值ϵ1=1e-7，物理损失阈值ϵ2=1e-6
    convergence_threshold_data = 1e-7  # 数据损失阈值
    convergence_threshold_phy = 1e-6   # 物理损失阈值
    convergence_patience = 50  # 联合收敛检查耐心
    convergence_counter = 0
    loss_change_data = 1.0
    loss_change_phy = 1.0
    converged = False
    epoch = 0

    # 物理约束
    # PINN权重 λ_phy，输入一个值
    input_str = input("请输入 PINN 权重 λ_phys (如0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0): ")
    lambda_phy = float(input_str) if input_str.strip() else 0.0
    print(f"PINN权重 λ_phys = {lambda_phy}")

    # 内部权重分配：λPDE=λbc=λenergy=λparam
    lambda_data = 1.0  # 数据损失权重固定为1
    # 物理损失内部权重（固定值，与λ_phy无关）
    lambda_pde = 1.0
    lambda_bc = 1.0
    lambda_energy = 0.1
    lambda_param = 1.0

    print(f"权重: λdata={lambda_data}, λPDE={lambda_pde}, λbc={lambda_bc}, λenergy={lambda_energy}, λparam={lambda_param}")

    # 初始化物理约束（需要前向模型作为"微分求解器"）
    # 默认路径：项目根目录下的 forward_model_best.pth（相对路径）
    forward_model_path = Path(__file__).parent / "forward_model_best.pth"

    if lambda_phy > 0 and Path(forward_model_path).exists():
        phy_constraint = InversePhysicsConstraint(
            device=device,
            w_pde=lambda_pde,
            w_bc=lambda_bc,
            w_energy=lambda_energy,
            w_param=lambda_param,
            forward_model_path=forward_model_path
        )
        use_physics = True
        print("物理约束已启用")
    else:
        phy_constraint = None
        use_physics = False
        if lambda_phy > 0 and not Path(forward_model_path).exists():
            print(f"物理约束已禁用：未找到前向模型权重 {forward_model_path}")
            print("  请先运行 `python 正向模型.py` 训练前向模型（会保存到上述路径）")
            print("  或编辑本脚本顶部的 forward_model_path 变量指向实际位置")
        else:
            print("物理约束已禁用（lambda_phys=0）")

    while True:
        epoch += 1
        model.train()
        train_loss = 0
        phy_loss = 0

        for batch_spectra, batch_params in train_loader:
            batch_spectra = batch_spectra.to(device)
            batch_params = batch_params.to(device)

            optimizer.zero_grad()

            # 前向: 光谱 -> 参数
            pred = model(batch_spectra)

            # 数据损失
            loss_data = criterion(pred, batch_params)
            loss = loss_data

            # 物理约束损失
            if use_physics:
                try:
                    phy_loss_val, _ = phy_constraint.compute_loss(pred)
                    if isinstance(phy_loss_val, torch.Tensor):
                        phy_loss += phy_loss_val.item()
                    else:
                        phy_loss += float(phy_loss_val)

                    # 使用固定权重 λ_phy
                    current_lambda_phy = lambda_phy

                    # ℒtotal = λdataℒdata + λphy(λPDEℒPDE + λbcℒbc + λenergyℒenergy + λparamℒparam)
                    loss = lambda_data * loss_data + current_lambda_phy * phy_loss_val
                except Exception as e:
                    if epoch <= 2:
                        print(f"  [物理约束警告] {e}")
                    loss = loss_data

            loss.backward()
            optimizer.step()

            train_loss += loss_data.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        phy_losses.append(phy_loss / len(train_loader) if phy_loss > 0 else 0)

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_spectra, batch_params in val_loader:
                batch_spectra = batch_spectra.to(device)
                batch_params = batch_params.to(device)
                pred = model(batch_spectra)
                val_loss += criterion(pred, batch_params).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step()

        # 检查联合收敛（数据损失和物理损失）
        if len(val_losses) >= 2 and len(phy_losses) >= 2:
            loss_change_data = abs(val_losses[-1] - val_losses[-2])
            loss_change_phy = abs(phy_losses[-1] - phy_losses[-2])
            # 联合收敛条件：两者都小于阈值
            if loss_change_data < convergence_threshold_data and loss_change_phy < convergence_threshold_phy:
                convergence_counter += 1
                if convergence_counter >= convergence_patience:
                    print(f"\n收敛: 连续{convergence_patience}轮数据损失变化 < {convergence_threshold_data} 且物理损失变化 < {convergence_threshold_phy}")
                    converged = True
                    break
            else:
                convergence_counter = 0

        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
            if use_physics and len(phy_losses) > 0:
                print(f"  | Phy: {phy_losses[-1]:.6f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = SAVE_DIR_PATH / 'inverse_model_best.pth'
            torch.save(model.state_dict(), save_path)
            print(f"  [保存最佳模型] Val loss: {val_loss:.6f}")
        else:
            patience_counter += 1
            # 最小训练轮数之前不触发早停
            if epoch >= min_epochs and patience_counter >= patience:
                print(f"\n早停: 连续{patience}轮未改善（已训练{epoch}轮）")
                break

    # 6. 保存最终模型
    save_path = SAVE_DIR_PATH / 'inverse_model_final.pth'
    torch.save(model.state_dict(), save_path)
    print(f"\n模型已保存: {save_path}")

    # 7. 测试集评估
    print("\n测试集评估...")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_spectra, batch_params in test_loader:
            batch_spectra = batch_spectra.to(device)
            pred = model(batch_spectra)
            all_preds.append(pred.cpu())
            all_targets.append(batch_params.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # 计算指标
    import numpy as np
    mse = nn.MSELoss()(all_preds, all_targets).item()
    rmse = np.sqrt(mse)
    mae = nn.L1Loss()(all_preds, all_targets).item()

    # MAPE（基于反归一化后的值，更有意义）
    # 重新计算，因为 preds_denorm 在下面定义

    # 分参数评估
    param_names = ['tilt_angle', 'delta_n', 'sigma']
    print(f"\n===== 测试集评估指标 =====")
    print(f"整体 MSE: {mse:.6f}")
    print(f"整体 RMSE: {rmse:.6f}")
    print(f"整体 MAE: {mae:.6f}")

    # 反归一化后评估
    preds_denorm = loader.denormalize_params(all_preds.numpy())
    targets_denorm = loader.denormalize_params(all_targets.numpy())

    # 计算整体MAPE（基于反归一化后的值）
    mape = np.mean(np.abs((preds_denorm - targets_denorm) / (targets_denorm + 1e-10))) * 100
    print(f"整体 MAPE: {mape:.4f}%")

    for i, name in enumerate(param_names):
        mse_p = nn.MSELoss()(all_preds[:, i], all_targets[:, i]).item()
        mae_p = nn.L1Loss()(all_preds[:, i], all_targets[:, i]).item()
        # 基于反归一化后的MAPE
        mape_p = np.mean(np.abs((preds_denorm[:, i] - targets_denorm[:, i]) / (targets_denorm[:, i] + 1e-10))) * 100
        print(f"{name}: MSE={mse_p:.6f}, MAE={mae_p:.6f}, MAPE={mape_p:.4f}%")

    print(f"\n反归一化后:")
    for i, name in enumerate(param_names):
        error = np.abs(preds_denorm[:, i] - targets_denorm[:, i])
        mean_error = error.mean()
        print(f"{name}: 平均误差 = {mean_error:.6e}")

    # 绘制损失曲线
    print("\n绘制损失曲线...")
    plt.figure(figsize=(18, 4))

    # 数据损失
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    plt.plot(val_losses, label='Val Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Data Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # PINN 物理损失
    plt.subplot(1, 3, 2)
    plt.plot(phy_losses, label='Physics Loss', color='orange', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PINN Physics Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 总损失（数据+物理）
    plt.subplot(1, 3, 3)
    total_losses = [d + p for d, p in zip(train_losses, phy_losses)]
    plt.plot(total_losses, label='Train Total', alpha=0.7)
    plt.plot(val_losses, label='Val Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Loss = Data + Physics')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(SAVE_DIR_PATH / 'loss_curve.png', dpi=150)
    print(f"损失曲线已保存: {SAVE_DIR_PATH / 'loss_curve.png'}")
    plt.close()

    # 保存结果
    np.savez(SAVE_DIR_PATH / 'training_history.npz',
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses),
        phy_losses=np.array(phy_losses))

    results = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'best_val_loss': best_val_loss,
        'epochs': epoch
    }
    np.savez(SAVE_DIR_PATH / 'results.npz', **results)

    # 保存预测结果（反归一化后，用于后续可视化）
    np.savez(SAVE_DIR_PATH / 'predictions.npz',
             pred=preds_denorm,
             true=targets_denorm)
    print(f"预测结果已保存: {SAVE_DIR_PATH / 'predictions.npz'}")

    # ========== 绘制可视化图表 ==========
    print("\n绘制可视化图表...")

    # 1. 参数预测散点图
    param_names = ['tilt_angle (deg)', 'delta_n (1e-4)', 'sigma (1e-4)']
    # 缩放以便显示
    pred_display = preds_denorm.copy()
    true_display = targets_denorm.copy()
    pred_display[:, 1] *= 1e4
    pred_display[:, 2] *= 1e4
    true_display[:, 1] *= 1e4
    true_display[:, 2] *= 1e4

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, (ax, name) in enumerate(zip(axes, param_names)):
        pred_i = pred_display[:, i]
        true_i = true_display[:, i]

        ax.scatter(true_i, pred_i, alpha=0.5, s=30, c='#2E86AB', edgecolors='none')

        # 对角线
        min_val = min(true_i.min(), pred_i.min())
        max_val = max(true_i.max(), pred_i.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Perfect Fit')

        # R²
        ss_res = np.sum((true_i - pred_i) ** 2)
        ss_tot = np.sum((true_i - np.mean(true_i)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        ax.set_xlabel(f'True {name}', fontsize=10)
        ax.set_ylabel(f'Predicted {name}', fontsize=10)
        ax.set_title(f'{name} (R2={r2:.4f})', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Inverse Model: Parameter Prediction Scatter Plot', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(SAVE_DIR_PATH / 'parameter_scatter.png', dpi=150, bbox_inches='tight')
    print(f"参数散点图已保存: {SAVE_DIR_PATH / 'parameter_scatter.png'}")
    plt.close()

    # 2. 误差分布直方图
    error = preds_denorm - targets_denorm

    param_names = ['tilt_angle (deg)', 'delta_n', 'sigma']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, (ax, name) in enumerate(zip(axes, param_names)):
        err = error[:, i]

        ax.hist(err, bins=20, color='#1B998B', alpha=0.7, edgecolor='white')
        ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
        ax.axvline(x=np.mean(err), color='red', linestyle='-', linewidth=1.5,
                  label=f'Mean={np.mean(err):.2e}')
        ax.set_xlabel('Prediction Error', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{name} Error', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Inverse Model: Parameter Error Distribution', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(SAVE_DIR_PATH / 'parameter_error_distribution.png', dpi=150, bbox_inches='tight')
    print(f"误差分布图已保存: {SAVE_DIR_PATH / 'parameter_error_distribution.png'}")
    plt.close()

    print("\n训练完成!")
    return model, results


if __name__ == "__main__":
    main_training()