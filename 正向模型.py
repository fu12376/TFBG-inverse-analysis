"""TFBG 光谱求逆 - 正向模型 (Forward Model)"""
# ============================================================================
# 第一部分：工具包加载
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

# 科学计算
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

# 可视化
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

# 配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 第二部分：数据加载
# ============================================================================

class TFBGDataLoader:
    """TFBG光谱数据加载器"""

    def __init__(self, data_dir):
        """
        初始化数据加载器

        Args:
            data_dir: 光谱数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.wavelength_range = (1520, 1600)  # nm
        self.spectrum_points = 4001

        # 固定参数（不参与回归）
        self.fixed_params = {
            'p': 1.5,        # 基底折射率
            'long': 14500     # 光栅长度
        }

        # 回归参数范围（根据实际数据统计）
        self.regress_params = {
            'tilt_angle': (7.0, 7.5),       # 光栅角度 (°) 实际范围
            'delta_n': (4.0e-4, 6.0e-4),    # 调制深度
            'sigma': (6.0e-4, 7.0e-4)     # 峰宽参数
        }

    def parse_filename(self, filename):
        """
        从文件名解析参数

        文件名格式: 波导结构层数为：四层p=1.5_光栅角度=7.1_光栅长度=14500_光栅周期=5.59e-01_调制深度=4.2e-04_sigma=6.4e-04.txt

        Returns:
            dict: 参数字典
        """
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
        """
        加载单个光谱文件

        Args:
            filepath: 文件路径

        Returns:
            numpy.ndarray: 光谱数据 (4001, 3) - [波长, 透射响应1, 透射响应2]
        """
        data = pd.read_csv(filepath, sep='\t', header=None)
        return data.values

    def load_all_data(self):
        """
        加载所有光谱数据

        Returns:
            dict: 包含以下键:
                - spectra: 光谱数据 (N, 4001, 2)
                - params: 参数数据 (N, 6) [tilt_angle, delta_n, sigma, period, p, long]
                - filenames: 文件名列表
        """
        txt_files = list(self.data_dir.glob('*.txt'))
        print(f"找到 {len(txt_files)} 个光谱文件")

        spectra = []
        params_list = []
        filenames = []

        for filepath in sorted(txt_files):
            try:
                # 加载光谱
                spectrum = self.load_spectrum(filepath)

                # 解析参数
                params = self.parse_filename(filepath.name)

                # 只保留需要回归的参数 + 周期
                param_array = np.array([
                    params['tilt_angle'],
                    params['delta_n'],
                    params['sigma'],
                    params['period'],
                    params['p'],
                    params['long']
                ])

                spectra.append(spectrum[:, 1:3])  # 透射响应1和透射响应2
                params_list.append(param_array)
                filenames.append(filepath.name)

            except Exception as e:
                print(f"加载失败 {filepath.name}: {e}")

        spectra = np.array(spectra)
        params_list = np.array(params_list)

        print(f"光谱数据形状: {spectra.shape}")
        print(f"参数数据形状: {params_list.shape}")

        return {
            'spectra': spectra,
            'params': params_list,
            'filenames': filenames
        }

    def normalize_spectra(self, spectra):
        """
        归一化光谱数据（每个样本单独归一化到 [-1, 1]）

        Args:
            spectra: 原始光谱 (N, 4001, 2)

        Returns:
            numpy.ndarray: 归一化后的光谱
        """
        spectra_norm = spectra.copy()

        # 透射响应1：dB值为负，用最小值的绝对值归一化
        for i in range(len(spectra)):
            v = spectra[i, :, 0]
            v_min = v.min()  # 最负的值
            if v_min < 0:
                spectra_norm[i, :, 0] = v / abs(v_min)
            else:
                spectra_norm[i, :, 0] = v

        # 透射响应2：归一化到 [-1, 1]
        for i in range(len(spectra)):
            v = spectra[i, :, 1]
            v_range = v.max() - v.min()
            if v_range > 0:
                # 归一化到 [-1, 1]
                spectra_norm[i, :, 1] = 2 * (v - v.min()) / v_range - 1
            else:
                spectra_norm[i, :, 1] = v

        return spectra_norm

    def normalize_params(self, params):
        """
        归一化参数到 [0, 1]

        Args:
            params: 原始参数 (N, 6)

        Returns:
            numpy.ndarray: 归一化后的参数
        """
        # tilt_angle: 7.0 - 7.5 实际范围
        params_norm = params.copy()
        params_norm[:, 0] = (params[:, 0] - 7.0) / 0.5

        # delta_n: 4.0e-4 - 6.0e-4
        params_norm[:, 1] = (params[:, 1] - 4.0e-4) / 2.0e-4

        # sigma: 6.0e-4 - 7.0e-4
        params_norm[:, 2] = (params[:, 2] - 6.0e-4) / 1.0e-4

        # period: 保持原值（不回归）
        params_norm[:, 3] = params[:, 3]

        # p: 固定值 1.5
        params_norm[:, 4] = (params[:, 4] - 1.5) / 1.5

        # long: 固定值 14500
        params_norm[:, 5] = (params[:, 5] - 14500) / 14500

        return params_norm

    def denormalize_params(self, params_norm):
        """
        反归一化参数

        Args:
            params_norm: 归一化后的参数 (N, 6)

        Returns:
            numpy.ndarray: 原始参数
        """
        params = params_norm.copy()

        # tilt_angle: 7.0 - 7.5
        params[:, 0] = params_norm[:, 0] * 0.5 + 7.0

        # delta_n
        params[:, 1] = params_norm[:, 1] * 2.0e-4 + 4.0e-4

        # sigma
        params[:, 2] = params_norm[:, 2] * 1.0e-4 + 6.0e-4

        return params

    def add_noise(self, spectra, noise_level=0.01):
        """
        添加噪声进行数据增广

        Args:
            spectra: 原始光谱
            noise_level: 噪声标准差

        Returns:
            numpy.ndarray: 加噪���的��谱
        """
        noise = np.random.randn(*spectra.shape) * noise_level
        return spectra + noise

    def split_data(self, data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        """
        划分训练/验证/测试集

        Args:
            data: 数据字典
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            seed: 随机种子

        Returns:
            tuple: (train_data, val_data, test_data)
        """
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
                'params': d['params'][indices],
                'filenames': [d['filenames'][i] for i in indices]
            }

        return (
            split_by_indices(data, train_idx),
            split_by_indices(data, val_idx),
            split_by_indices(data, test_idx)
        )

    def save_data(self, data, save_path):
        """
        保存数据为 npz 格式

        Args:
            data: 数据字典
            save_path: 保存路径
        """
        np.savez(
            save_path,
            spectra=data['spectra'],
            params=data['params'],
            filenames=np.array(data['filenames'])
        )
        print(f"数据已保存到: {save_path}")


class TFBGDataset(Dataset):
    """TFBG 光谱数据集"""

    def __init__(self, spectra, params):
        """
        初始化数据集

        Args:
            spectra: 光谱数据 (N, 4001, 2)
            params: 参数数据 (N, 6)
        """
        self.spectra = torch.FloatTensor(spectra)
        self.params = torch.FloatTensor(params)

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        return self.spectra[idx], self.params[idx]


# 测试数据加载
if __name__ == "__main__":
    # 设置路径
    DATA_DIR = Path(__file__).parent / "data"

    # 创建数据加载器
    loader = TFBGDataLoader(DATA_DIR)

    # 加载所有数据
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
        'params': params_norm,
        'filenames': data['filenames']
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

    # 检查原始光谱数据分布
    print(f"\n原始光谱数据分布:")
    print(f"透射响应1 - min: {data['spectra'][:,:,0].min():.3f}, max: {data['spectra'][:,:,0].max():.3f}")
    print(f"透射响应2 - min: {data['spectra'][:,:,1].min():.3f}, max: {data['spectra'][:,:,1].max():.3f}")
    print(f"透射响应1 abs max: {np.abs(data['spectra'][:,:,0]).max():.3f}")
    print(f"透射响应2 abs max: {np.abs(data['spectra'][:,:,1]).max():.3f}")


# ============================================================================
# 第三部分：1dCNN神经网络（CNN + 全连接 + 残差模块）
# ============================================================================

class SinActivation(nn.Module):
    """Sin 激活函数（参考PINN）"""
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

        # 残差连接
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


class ConvBlock(nn.Module):
    """卷积块 + 残差"""
    def __init__(self, in_channels, out_channels, kernel_size=16, stride=4):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = SinActivation()
        self.residual = ResidualBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.residual(x)
        return x


class ForwardNet(nn.Module):
    """正向网络：参数 → 光谱

    输入: 3个参数 (tilt_angle, delta_n, sigma)
    输出: 光谱 (4001, 2)
    """

    def __init__(self, input_dim=3, output_len=4001, output_channels=2):
        super().__init__()

        self.output_len = output_len
        self.output_channels = output_channels

        # 输入参数编码
        self.param_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            SinActivation(),
            nn.Linear(64, 128),
            SinActivation(),
            nn.Linear(128, 256),
            SinActivation(),
        )

        # 将参数编码扩展为序列
        # 方法：先映射到特征维度，再reshape为序列
        self.feature_expand = nn.Sequential(
            nn.Linear(256, 512),
            SinActivation(),
            nn.Linear(512, 1024),
            SinActivation(),
        )

        # 1D CNN 处理序列特征
        self.cnn = nn.Sequential(
            # 将特征作为单通道序列处理
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            SinActivation(),
            ResidualBlock(64, 64),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            SinActivation(),
            ResidualBlock(128, 128),

            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            SinActivation(),
            ResidualBlock(256, 256),

            nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            SinActivation(),
        )

        # 全连接层输出光谱
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            SinActivation(),

            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            SinActivation(),
            nn.Dropout(0.2),
            nn.Linear(1024, output_len * output_channels),
        )

    def forward(self, x):
        """
        Args:
            x: 输入参数 (batch, 3) - [tilt_angle, delta_n, sigma]

        Returns:
            光谱 (batch, output_len, 2)
        """
        batch_size = x.size(0)

        x = self.param_encoder(x)  # (batch, 256)
        x = self.feature_expand(x)  # (batch, 1024)

        # 重塑为序列 (batch, 1, 1024)
        x = x.view(batch_size, 1, -1)

        x = self.cnn(x)  # (batch, 256, N)

        x = torch.mean(x, dim=2)  # (batch, 256)

        x = self.fc(x)  # (batch, 4001*2)

        x = x.view(batch_size, self.output_len, self.output_channels)

        return x


# ============================================================================
# 第四部分：物理约束（自包含实现，不依赖 物理模型/physics_model.py）
# ============================================================================
#
# 物理约束完全在本文件中实现，含 3 项子损失：透射响应1 范围 / 光谱平滑 / 透射响应2 范围。


class PhysicsConstraint:
    """
    物理约束封装（自包含）
    直接对网络输出施加物理先验约束，无需任何外部物理求解器
    """

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化物理约束

        Args:
            device: 计算设备
        """
        self.device = torch.device(device)

    def compute_loss(self, spectrum_pred):
        """
        计算物理约束 loss（保持梯度反传，用于 PINN 训练）

        Args:
            spectrum_pred: 预测光谱 (batch, 4001, 2)

        Returns:
            physics_loss: 物理约束总loss（tensor，保持计算图）
            loss_dict: 各分项loss（Python float，仅用于记录）
        """
        # 物理约束（保持 tensor 累加，确保梯度可反传）
        loss_dict = {}

        # 1. ℒ_bc_s: 透射响应1 范围约束（非对称 - 物理上 dB 透射值必须 ≤0）
        # 只惩罚 S > 0 的部分（透射超过 0 dB 是非物理的）
        S = spectrum_pred[:, :, 0]  # (batch, 4001)
        S_violation = torch.relu(S)  # 非对称：max(0, S)
        loss_bc_s = S_violation.mean()
        loss_dict['bc_s'] = loss_bc_s.item()

        # 2. ℒ_pde: 光谱平滑性约束（S 和 P 两个通道相邻点差值 ≤ 0.1 dB）
        S_diff = S[:, 1:] - S[:, :-1]
        S_diff_violation = torch.relu(torch.abs(S_diff) - 0.1)
        P_diff_violation = torch.zeros_like(S_diff_violation)
        if spectrum_pred.shape[2] > 1:
            P = spectrum_pred[:, :, 1]
            P_diff = P[:, 1:] - P[:, :-1]
            P_diff_violation = torch.relu(torch.abs(P_diff) - 0.1)
        loss_pde = (S_diff_violation.mean() + P_diff_violation.mean()) * 0.5
        loss_dict['pde'] = loss_pde.item()

        # 3. ℒ_bc_p: 透射响应2 范围约束（归一化后应在 [-1, 1]）
        if spectrum_pred.shape[2] > 1:
            P = spectrum_pred[:, :, 1]
            P_violation = torch.relu(torch.abs(P) - 1.0)
            loss_bc_p = P_violation.mean()
            loss_dict['bc_p'] = loss_bc_p.item()
            total_loss = loss_bc_s + loss_pde + loss_bc_p
        else:
            total_loss = loss_bc_s + loss_pde

        # 保存总损失（仅用于记录）
        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict

    def compute_forward_loss(self, spectrum_pred, spectrum_true):
        """
        计算前向模型的总loss（数据 + 物理）

        Args:
            spectrum_pred: 预测光谱 (batch, 4001, 2)
            spectrum_true: 真实光谱 (batch, 4001, 2)

        Returns:
            total_loss: 总loss
            loss_dict: 各分项loss
        """
        criterion = nn.MSELoss()

        # 数据loss
        data_loss = criterion(spectrum_pred, spectrum_true)

        # 物理loss
        physics_loss, physics_dict = self.compute_loss(spectrum_pred)

        # 总loss
        total_loss = data_loss + physics_loss

        loss_dict = {
            'data': data_loss.item(),
            'bc_s': physics_dict.get('bc_s', 0),
            'pde': physics_dict.get('pde', 0),
            'bc_p': physics_dict.get('bc_p', 0),
            'physics': physics_loss.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict

    def compute_inverse_loss(self, params_pred, params_true, spectrum_pred=None):
        """
        计算逆向模型的总loss（数据 + 物理）

        Args:
            params_pred: 预测参数 (batch, 3)
            params_true: 真实参数 (batch, 3)
            spectrum_pred: 预测光谱，可选，用于物理约束

        Returns:
            total_loss: 总loss
            loss_dict: 各分项loss
        """
        criterion = nn.MSELoss()

        # 数据loss（参数）
        data_loss = criterion(params_pred, params_true)

        # 物理约束loss（如果提供了光谱）
        physics_loss = data_loss.new_zeros(())
        physics_dict = {}
        if spectrum_pred is not None:
            physics_loss, physics_dict = self.compute_loss(spectrum_pred)

        # 总loss
        total_loss = data_loss + physics_loss

        loss_dict = {
            'data': data_loss.item(),
            'params': data_loss.item(),
            'physics': physics_loss.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict


# ============================================================================
# 第五部分：输出和保存
# ============================================================================

class ModelExporter:
    """模型导出和保存"""

    def __init__(self, save_dir='./output'):
        """
        初始化导出器

        Args:
            save_dir: 保存目录
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

    def save_model(self, model, filename, metadata=None):
        """
        保存模型

        Args:
            model: PyTorch模型
            filename: 文件名
            metadata: 元数据（可选）
        """
        save_path = self.save_dir / filename

        # 保存模型权重
        torch.save(model.state_dict(), save_path)

        # 保存元数据
        if metadata is not None:
            meta_path = save_path.replace('.pth', '_meta.npz')
            np.savez(meta_path, **metadata)

        print(f"模型已保存: {save_path}")

        return save_path

    def load_model(self, model_class, filename, **kwargs):
        """
        加载模型

        Args:
            model_class: 模型类
            filename: 文件名
            **kwargs: 模型初始化参数

        Returns:
            loaded_model: 加载后的模型
        """
        save_path = self.save_dir / filename

        model = model_class(**kwargs)
        model.load_state_dict(torch.load(save_path))
        model.eval()

        print(f"模型已加载: {save_path}")

        return model

    def save_results(self, results, filename):
        """
        保存预测结果

        Args:
            results: 预测结果字典
            filename: 文件名
        """
        save_path = self.save_dir / filename

        # 转换为numpy并保存
        np.savez(
            save_path,
            **{k: v.cpu().numpy() if torch.is_tensor(v) else v
               for k, v in results.items()}
        )

        print(f"结果已保存: {save_path}")

        return save_path

    def plot_spectrum(self, wavelength, spectrum_pred, spectrum_true=None,
                      save_path=None, title='Spectrum Comparison'):
        """
        绘制光谱对比图

        Args:
            wavelength: 波长数组
            spectrum_pred: 预测光谱
            spectrum_true: 真实光谱（可选）
            save_path: 保存路径
            title: 标题
        """
        plt.figure(figsize=(12, 4))

        # 透射响应1
        plt.subplot(1, 2, 1)
        plt.plot(wavelength, spectrum_pred[:, 0], 'b-', label='Pred', alpha=0.7)
        if spectrum_true is not None:
            plt.plot(wavelength, spectrum_true[:, 0], 'r--', label='True', alpha=0.7)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Transmission 1 (dB)')
        plt.title(title + ' - Transmission 1')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 透射响应2
        plt.subplot(1, 2, 2)
        plt.plot(wavelength, spectrum_pred[:, 1], 'b-', label='Pred', alpha=0.7)
        if spectrum_true is not None:
            plt.plot(wavelength, spectrum_true[:, 1], 'r--', label='True', alpha=0.7)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Transmission 2')
        plt.title(title + ' - Transmission 2')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"图像已保存: {save_path}")

        plt.close()

    def plot_params_comparison(self, params_pred, params_true, param_names=None,
                            save_path=None):
        """
        绘制参数对比图

        Args:
            params_pred: 预测参数
            params_true: 真实参数
            param_names: 参数名
            save_path: 保存路径
        """
        if param_names is None:
            param_names = ['tilt_angle', 'delta_n', 'sigma']
        if params_true is None:
            param_names = param_names[:len(params_pred[0])]

        n_params = len(param_names)
        fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))

        if n_params == 1:
            axes = [axes]

        for i, (ax, name) in enumerate(zip(axes, param_names)):
            ax.scatter(params_true[:, i], params_pred[:, i], alpha=0.5)
            # 理想线
            min_val = min(params_true[:, i].min(), params_pred[:, i].min())
            max_val = max(params_true[:, i].max(), params_pred[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
            ax.set_xlabel(f'True {name}')
            ax.set_ylabel(f'Pred {name}')
            ax.set_title(f'{name} Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"图像已保存: {save_path}")

        plt.close()

    def plot_loss_curve(self, train_losses, val_losses=None, save_path=None):
        """
        绘制损失曲线

        Args:
            train_losses: 训练损失
            val_losses: 验证损失
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train')
        if val_losses is not None:
            plt.plot(val_losses, label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(train_losses, label='Train')
        if val_losses is not None:
            plt.plot(val_losses, label='Val')
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log)')
        plt.title('Training Loss Curve (log scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"图像已保存: {save_path}")

        plt.close()

    def export_summary(self, model_info, metrics, filename='summary.txt'):
        """
        导出训练摘要

        Args:
            model_info: 模型信息
            metrics: 评估指标
            filename: 文件名
        """
        save_path = self.save_dir / filename

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("TFBG 光谱求逆 - 模型训练摘要\n")
            f.write("=" * 50 + "\n\n")

            f.write("模型信息:\n")
            f.write("-" * 30 + "\n")
            for k, v in model_info.items():
                f.write(f"  {k}: {v}\n")

            f.write("\n评估指标:\n")
            f.write("-" * 30 + "\n")
            for k, v in metrics.items():
                f.write(f"  {k}: {v}\n")

        print(f"摘要已保存: {save_path}")

        return save_path


class Trainer:
    """训练器封装"""

    def __init__(self, model, physics_constraint=None, device='cuda'):
        """
        初始化训练器

        Args:
            model: 模型
            physics_constraint: 物理约束
            device: 设备
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.phy_constraint = physics_constraint
        self.exporter = ModelExporter()

        # 训练记录
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader, optimizer, criterion):
        """
        训练一个epoch

        Args:
            train_loader: 训练数据加载器
            optimizer: 优化器
            criterion: 损失函数

        Returns:
            avg_loss: 平均损失
        """
        self.model.train()
        total_loss = 0
        n = 0

        for batch in train_loader:
            spectra, params = batch
            spectra = spectra.to(self.device)
            params = params.to(self.device)

            optimizer.zero_grad()

            # 前向传播
            pred = self.model(params)

            # 数据loss
            loss = criterion(pred, spectra)

            # 物理loss（如果有）
            if self.phy_constraint is not None:
                phy_loss, _ = self.phy_constraint.compute_loss(pred)
                loss = loss + phy_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n += 1

        return total_loss / n

    def validate(self, val_loader, criterion):
        """
        验证

        Args:
            val_loader: 验证数据加载器
            criterion: 损失函数

        Returns:
            avg_loss: 平均损失
        """
        self.model.eval()
        total_loss = 0
        n = 0

        with torch.no_grad():
            for batch in val_loader:
                spectra, params = batch
                spectra = spectra.to(self.device)
                params = params.to(self.device)

                pred = self.model(params)
                loss = criterion(pred, spectra)

                total_loss += loss.item()
                n += 1

        return total_loss / n

    def fit(self, train_loader, val_loader, epochs=100, lr=0.001,
           w_phy=1.0, early_stopping_patience=20, save_best=True):
        """
        训练模型

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            lr: 学习率
            w_phy: 物理loss权重
            early_stopping_patience: 早停耐心值
            save_best: 是否保存最优模型

        Returns:
            history: 训练历史
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train': [], 'val': []}

        print(f"\n开始训练...")
        print(f"  epochs: {epochs}")
        print(f"  learning rate: {lr}")
        print(f"  physics weight: {w_phy}")

        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss = self.validate(val_loader, criterion)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            history['train'].append(train_loss)
            history['val'].append(val_loss)

            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_best:
                    self.exporter.save_model(
                        self.model,
                        'forward_model_best.pth',
                        {'epoch': epoch, 'val_loss': val_loss}
                    )
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\n早停: 验证损失连续{early_stopping_patience}轮未改善")
                    break

        print(f"\n训练完成! 最佳验证损失: {best_val_loss:.4f}")

        return history


# ============================================================================
# 主训练入口
# ============================================================================

def main_training():
    """主训练函数"""
    print("=" * 60)
    print("TFBG 光谱求逆 - 正向模型训练")
    print("=" * 60)

    # 1. 设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    DATA_DIR = Path(__file__).parent / "data"
    SAVE_DIR = Path(__file__).parent / "code-forward" / "output"
    SAVE_DIR_PATH = Path(SAVE_DIR)
    SAVE_DIR_PATH.mkdir(parents=True, exist_ok=True)

    # 2. 加载数据
    print("\n加载数据...")
    loader = TFBGDataLoader(DATA_DIR)
    data = loader.load_all_data()

    # 归一化
    spectra_norm = loader.normalize_spectra(data['spectra'])
    params_norm = loader.normalize_params(data['params'])

    # 划分数据集
    train_data, val_data, test_data = loader.split_data({
        'spectra': spectra_norm,
        'params': params_norm,
        'filenames': data['filenames']
    }, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    # 创建 DataLoader
    # 注意：只取前3列（tilt_angle, delta_n, sigma）作为输入
    train_dataset = TFBGDataset(train_data['spectra'], train_data['params'][:, :3])
    val_dataset = TFBGDataset(val_data['spectra'], val_data['params'][:, :3])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    print(f"训练集: {len(train_dataset)} 样本, {len(train_loader)} batches")
    print(f"验证集: {len(val_dataset)} 样本, {len(val_loader)} batches")

    # 3. 初始化模型
    print("\n初始化模型...")
    model = ForwardNet(input_dim=3, output_len=4001, output_channels=2).to(device)

    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 4. 训练设置
    criterion = nn.MSELoss()
    # 使用更小的学习率和权重衰减（防止过拟合）
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7)

    # 5. 训练循环（早停控制，无epoch限制）
    print("\n开始训练...")
    best_val_loss = float('inf')
    patience = 30  # 早停耐心值（验证损失连续30轮无改善则停止）
    min_epochs = 100  # 最小训练轮数（在此之前不触发早停）
    patience_counter = 0
    train_losses = []
    val_losses = []

    # 收敛判断：最小改进阈值1e-6
    convergence_threshold = 1e-6
    convergence_patience = 5  # 连续5次变化小于阈值则认为收敛
    convergence_counter = 0  # 收敛计数器
    loss_change = 1.0  # 初始值
    converged = False  # 收敛标志
    epoch = 0  # 当前epoch

    # 物理损失记录
    phy_losses = []

    # 尝试初始化物理约束
    try:
        phy_constraint = PhysicsConstraint(device=device)
        use_physics = True
        print("物理约束已启用 (PINN)")
    except Exception as e:
        print(f"物理约束初始化失败: {e}，使用简化版本")
        use_physics = False
        phy_constraint = None

    # PINN权重 lambda_phys，可交互式输入
    # 如果物理约束初始化失败，直接禁用PINN
    input_str = input("请输入PINN权重 lambda_phys (如0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0): ")
    lambda_phys = float(input_str) if input_str.strip() else 0
    print(f"PINN权重 lambda_phys = {lambda_phys}")

    # 动态权重调度：λphys(t) = λ0 * (1 - e^(-k*t))
    # 设置 λ0=1.0, k=0.01
    use_dynamic_lambda = False
    if lambda_phys > 0 and phy_constraint is not None:
        use_dynamic_input = input("是否使用动态权重调度? (y/n): ").strip().lower()
        if use_dynamic_input == 'y':
            use_dynamic_lambda = True
            lambda_0 = 1.0  # 最大权重
            k = 0.01  # 增长速率系数
            print(f"动态权重调度已启用: λ0={lambda_0}, k={k}")

    # 如果lambda_phys=0或物理约束初始化失败，禁用PINN
    if lambda_phys == 0 or phy_constraint is None:
        use_physics = False
        print("PINN已禁用")

    # 训练循环（无epoch限制，收敛或早停停止）
    print("\n开始训练...")

    while True:
        epoch += 1
        train_loss = 0
        phy_loss = 0
        for batch_spectra, batch_params in train_loader:
            batch_spectra = batch_spectra.to(device)
            batch_params = batch_params.to(device)

            optimizer.zero_grad()

            # 前向: 参数 -> 光谱
            pred = model(batch_params)

            # 数据损失
            loss_data = criterion(pred, batch_spectra)
            loss = loss_data

            # 物理约束损失 (PINN)
            if use_physics and phy_constraint is not None:
                try:
                    phy_loss_val, loss_dict = phy_constraint.compute_loss(pred)
                    # 累加物理损失 - 获取实际数值
                    if isinstance(phy_loss_val, torch.Tensor):
                        phy_loss += phy_loss_val.item()
                    else:
                        phy_loss += float(phy_loss_val)

                    # 动态权重调度: λphys(t) = λ0 * (1 - e^(-k*t))
                    if use_dynamic_lambda:
                        current_lambda = lambda_0 * (1 - np.exp(-k * epoch))
                    else:
                        current_lambda = lambda_phys

                    loss = loss_data + current_lambda * phy_loss_val
                except Exception as e:
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
                pred = model(batch_params)
                val_loss += criterion(pred, batch_spectra).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        # 检查收敛（基于验证损失变化）
        if len(val_losses) >= 2:
            loss_change = abs(val_losses[-1] - val_losses[-2])
            if loss_change < convergence_threshold:
                convergence_counter += 1
                if convergence_counter >= convergence_patience:
                    print(f"\n收敛: 连续{convergence_patience}轮损失变化 < {convergence_threshold}")
                    print(f"  最终训练损失: {train_loss:.6f}")
                    print(f"  最终验证损失: {val_loss:.6f}")
                    converged = True
                    break
            else:
                convergence_counter = 0

        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Δ: {loss_change:.2e}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = SAVE_DIR_PATH / 'forward_model_best.pth'
            torch.save(model.state_dict(), save_path)
            print(f"  [保存最佳模型] Val loss: {val_loss:.6f}")
        else:
            patience_counter += 1
            # 最小训练轮数之前不触发早停
            if epoch >= min_epochs and patience_counter >= patience:
                print(f"\n早停: 连续{patience}轮未改善（已训练{epoch}轮）")
                break

    # 6. 保存最终模型
    save_path = SAVE_DIR_PATH / 'forward_model_final.pth'
    torch.save(model.state_dict(), save_path)
    print(f"\n模型已保存: {save_path}")

    # 保存训练历史
    np.savez(
        SAVE_DIR_PATH / 'training_history.npz',
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses)
    )
    print(f"训练历史已保存: {SAVE_DIR_PATH / 'training_history.npz'}")

    # 7. 测试集评估（多指标）
    print("\n测试集评估...")
    test_dataset = TFBGDataset(test_data['spectra'], test_data['params'][:, :3])
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_spectra, batch_params in test_loader:
            batch_spectra = batch_spectra.to(device)
            batch_params = batch_params.to(device)
            pred = model(batch_params)
            all_preds.append(pred.cpu())
            all_targets.append(batch_spectra.cpu())

    # 合并所有预测
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # 计算各项指标
    # 整体 MSE, RMSE, MAE
    mse = nn.MSELoss()(all_preds, all_targets).item()
    rmse = np.sqrt(mse)
    mae = nn.L1Loss()(all_preds, all_targets).item()

    # 分通道评估
    mse_trans = nn.MSELoss()(all_preds[:,:,0], all_targets[:,:,0]).item()
    mse_trans2 = nn.MSELoss()(all_preds[:,:,1], all_targets[:,:,1]).item()
    rmse_trans = np.sqrt(mse_trans)
    rmse_trans2 = np.sqrt(mse_trans2)
    mae_trans = nn.L1Loss()(all_preds[:,:,0], all_targets[:,:,0]).item()
    mae_trans2 = nn.L1Loss()(all_preds[:,:,1], all_targets[:,:,1]).item()

    print(f"\n===== 测试集评估指标 =====")
    print(f"整体:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"\n透射响应1 (通道1):")
    print(f"  MSE:  {mse_trans:.6f}")
    print(f"  RMSE: {rmse_trans:.6f}")
    print(f"  MAE:  {mae_trans:.6f}")
    print(f"\n透射响应2 (通道2):")
    print(f"  MSE:  {mse_trans2:.6f}")
    print(f"  RMSE: {rmse_trans2:.6f}")
    print(f"  MAE:  {mae_trans2:.6f}")

    # 保存测试结果
    results = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mse_trans': mse_trans,
        'mse_trans2': mse_trans2,
        'mae_trans': mae_trans,
        'mae_trans2': mae_trans2,
        'best_val_loss': best_val_loss,
        'epochs': epoch + 1
    }
    np.savez(SAVE_DIR_PATH / 'results.npz', **results)

    # 保存预测结果（用于后续可视化）
    np.savez(SAVE_DIR_PATH / 'predictions.npz',
             pred=all_preds.numpy(),
             true=all_targets.numpy(),
             params=loader.denormalize_params(test_data['params'][:, :3]))  # (N, 3): 原始参数
    print(f"预测结果已保存: {SAVE_DIR_PATH / 'predictions.npz'}")

    # ========== 绘制可视化图表 ==========
    print("\n绘制可视化图表...")

    # 生成波长数组
    wavelength = np.linspace(1520, 1600, 4001)

    # 1. 光谱预测对比图（仅选取2个典型样本，用于论文）
    test_params = loader.denormalize_params(test_data['params'][:, :3])  # 反归一化为原始参数
    sample_indices = [0, min(9, all_preds.shape[0]-1)]  # 选取前10%和最后10%的样本

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes = axes.flatten()

    for i, idx in enumerate(sample_indices):
        ax = axes[i]
        ax.plot(wavelength, all_targets[idx, :, 0].numpy(), 'r-', label='True', alpha=0.8, linewidth=1.2)
        ax.plot(wavelength, all_preds[idx, :, 0].numpy(), 'b--', label='Pred', alpha=0.8, linewidth=1.2)
        ax.set_xlabel('Wavelength (nm)', fontsize=11)
        ax.set_ylabel('Transmission 1 (dB)', fontsize=11)
        ax.set_title(f'Sample {i+1}', fontsize=11)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1520, 1600)

    # 在图下方添加参数信息
    fig.text(0.5, 0.02, f'Sample 1: tilt_angle={test_params[sample_indices[0], 0]:.2f}°, delta_n={test_params[sample_indices[0], 1]:.2e}, σ={test_params[sample_indices[0], 2]:.2e}   |   '
                           f'Sample 2: tilt_angle={test_params[sample_indices[1], 0]:.2f}°, delta_n={test_params[sample_indices[1], 1]:.2e}, σ={test_params[sample_indices[1], 2]:.2e}',
             ha='center', va='bottom', fontsize=10)

    plt.suptitle('Forward Model: Spectrum Prediction vs Ground Truth', fontsize=13, y=1.02)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(SAVE_DIR_PATH / 'spectrum_comparison.png', dpi=300, bbox_inches='tight')
    print(f"光谱对比图已保存: {SAVE_DIR_PATH / 'spectrum_comparison.png'}")
    plt.close()

    # 2. 误差分布直方图
    error_trans = (all_preds[:, :, 0] - all_targets[:, :, 0]).numpy().flatten()
    error_trans2 = (all_preds[:, :, 1] - all_targets[:, :, 1]).numpy().flatten()

    # 计算横轴范围（使用百分位数）
    trans_min, trans_max = np.percentile(error_trans, [1, 99])
    trans2_min, trans2_max = np.percentile(error_trans2, [1, 99])

    # 计算更精确的统计量
    mean_trans = np.mean(error_trans)
    mean_trans2 = np.mean(error_trans2)
    std_trans = np.std(error_trans)
    std_trans2 = np.std(error_trans2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 透射响应1误差
    ax = axes[0]
    ax.hist(error_trans, bins=50, color='#1B998B', alpha=0.7, edgecolor='white', range=(trans_min, trans_max))
    ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
    ax.axvline(x=mean_trans, color='red', linestyle='-', linewidth=1.5,
              label=f'Mean={mean_trans:.2e}, Std={std_trans:.2e}')
    ax.set_xlabel('Prediction Error (dB)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Transmission 1 Error Distribution', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 透射响应2误差
    ax = axes[1]
    ax.hist(error_trans2, bins=50, color='#1B998B', alpha=0.7, edgecolor='white', range=(trans2_min, trans2_max))
    ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
    ax.axvline(x=mean_trans2, color='red', linestyle='-', linewidth=1.5,
              label=f'Mean={mean_trans2:.2e}, Std={std_trans2:.2e}')
    ax.set_xlabel('Prediction Error', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Transmission 2 Error Distribution', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Forward Model: Error Distribution', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(SAVE_DIR_PATH / 'error_distribution.png', dpi=300, bbox_inches='tight')
    print(f"误差分布图已保存: {SAVE_DIR_PATH / 'error_distribution.png'}")

    plt.close()

    # 绘制损失曲线（包含 PINN 物理损失）
    print("\n绘制损失曲线...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # 数据损失
    axes[0].plot(train_losses, label='Train', alpha=0.7)
    axes[0].plot(val_losses, label='Val', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Data Loss (MSE)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # PINN 物理损失
    if phy_losses:
        axes[1].plot(phy_losses, label='Physics Loss', color='orange', alpha=0.7)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('PINN Physics Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'Physics Loss\nNot Available', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('PINN Physics Loss')

    # 总损失 (数据 + 物理)
    if phy_losses:
        total_train = [d + p for d, p in zip(train_losses, phy_losses)]
        axes[2].plot(total_train, label='Train Total', alpha=0.7)
        axes[2].plot(val_losses, label='Val (Data)', alpha=0.7)
    else:
        axes[2].plot(train_losses, label='Train', alpha=0.7)
        axes[2].plot(val_losses, label='Val', alpha=0.7)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Total Loss (Data + Physics)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(SAVE_DIR_PATH / 'loss_curve.png', dpi=150)
    print(f"损失曲线已保存: {SAVE_DIR_PATH / 'loss_curve.png'}")
    plt.close()

    # 保存所有损失到文件
    save_data = {
        'train_losses': np.array(train_losses),
        'val_losses': np.array(val_losses)
    }
    if phy_losses:
        save_data['phy_losses'] = np.array(phy_losses)
    np.savez(SAVE_DIR_PATH / 'training_history.npz', **save_data)
    print(f"训练历史已保存: {SAVE_DIR_PATH / 'training_history.npz'}")

    return model, results


if __name__ == "__main__":
    main_training()