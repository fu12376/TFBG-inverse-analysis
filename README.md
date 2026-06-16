# TFBG 光谱逆源分析（PINN）

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: Research](https://img.shields.io/badge/License-Research%20Use-lightgrey.svg)](#许可)

基于物理约束神经网络（Physics-Informed Neural Network, PINN）的倾斜光纤布拉格光栅（TFBG）光谱求逆研究项目。

## 项目简介

TFBG 的透射谱与栅格参数之间存在强物理关系（耦合模理论），本项目用深度学习从数据中拟合这一映射：

- **正向模型**（参数 → 光谱）：基于 1D-CNN + SIREN（sin 激活）的端到端回归器，将 3 个光栅参数 `(tilt_angle, delta_n, sigma)` 回归为 4001×2 的透射谱
- **逆向模型**（光谱 → 参数）：基于 1D-CNN + 残差块的回归器，从透射谱反推 3 个光栅参数
- **物理约束（PINN）**：将"dB 透射值非正"、"光谱平滑性"等先验嵌入训练目标，提升泛化能力

## 目录结构

```
TFBG光谱逆源分析/
├── 正向模型.py                # 正向训练脚本（参数 → 光谱）
├── 逆向模型.py                # 逆向训练脚本（光谱 → 参数）
├── code-forward/output/       # 正向训练产物：权重 + 损失曲线 + 光谱对比图
├── code-inverse/output/       # 逆向训练产物：同上
├── .gitignore
└── README.md
```

> **数据集未包含在本仓库中**。`data/` 目录（379 个 TFBG 光谱 `.txt`，~48 MB）由第三方仿真生成，**仅供作者本地使用**。如需复现实验，请通过邮件联系作者获取；或自行按下方"数据格式"说明用耦合模理论仿真生成。

## 环境依赖

| 包 | 版本建议 | 说明 |
|---|---|---|
| Python | 3.9 / 3.12 | |
| PyTorch | 2.0+ | CUDA 12.x 推荐；纯 CPU 也可跑 |
| numpy | 1.24+ | |
| pandas | 2.0+ | 读取 `.txt` 光谱 |
| matplotlib | 3.7+ | 训练曲线 / 光谱对比图 |
| scipy | 1.10+ | 可选的信号平滑 |

安装：

```bash
pip install torch numpy pandas matplotlib scipy
```

> ⚠️ matplotlib 默认尝试加载 SimHei 渲染中文，如系统未装该字体，标题/标签会显示为方块。改用英文标签，或安装 SimHei 即可。

## 数据格式

> 以下说明用于**自行准备数据集**时的格式参考。本仓库不包含数据。

`data/` 下每个 `.txt` 文件命名格式：

```
波导结构层数为：四层p=1.5_光栅角度=7.1_光栅长度=14500_光栅周期=5.59e-01_调制深度=4.2e-04_sigma=6.4e-04.txt
```

文件内容：4001 行 × 3 列（制表符分隔）—— `[波长(nm), 透射响应1(dB), 透射响应2]`。训练脚本会自动从文件名解析出 3 个待回归参数 `(tilt_angle, delta_n, sigma)`，其余视为常量（`p=1.5`, `long=14500`, `period` 仅作参考不参与回归）。

放入 `data/` 后即可直接运行训练脚本。

## 运行方式

### 正向训练

```bash
python 正向模型.py
```

脚本会提示输入 PINN 权重 `lambda_phys`（建议先填 `0` 跑纯数据基线）。训练完成后 `code-forward/output/` 下生成：

- `forward_model_best.pth` / `forward_model_final.pth` — 模型权重
- `loss_curve.png` — 训练 / 验证损失 + 物理损失曲线
- `spectrum_comparison.png` — 透射响应1 预测 vs 真实
- `error_distribution.png` — 双通道误差直方图
- `predictions.npz` / `results.npz` / `training_history.npz` — 数值化产物

### 逆向训练

```bash
python 逆向模型.py
```

依赖正向模型作为"微分求解器"（用于物理约束中由参数反算光谱）。脚本顶部 `forward_model_path` 需指向 `code-forward/output/forward_model_best.pth`，若路径不同请先修改。

## 网络架构

### 正向网络 `ForwardNet`（`正向模型.py` 第 3 部分）

`3 个参数 → 4001×2 光谱`，由 5 个子模块串联：

```
输入(3)
  ↓ Linear(3,64) → SIREN → Linear(64,128) → SIREN → Linear(128,256)         [param_encoder]
  ↓ Linear(256,512) → SIREN → Linear(512,1024)                              [feature_expand]
  ↓ reshape → (B, 1, 1024)
  ↓ 1D-CNN: Conv1d(1→64,k7,s2) → BN → SIREN → ResBlock(64)
        → Conv1d(64→128,k5,s2) → BN → SIREN → ResBlock(128)
        → Conv1d(128→256,k5,s2) → BN → SIREN → ResBlock(256)
        → Conv1d(256→256,k3,s2) → BN → SIREN                                [cnn]
  ↓ mean(dim=2)  → (B, 256)
  ↓ Linear(256,512) → SIREN → Dropout(0.2)
        → Linear(512,1024) → SIREN → Dropout(0.2)
        → Linear(1024, 8002)                                                 [fc]
  ↓ reshape → (B, 4001, 2)
```

设计要点：
- **SIREN 激活**（`sin(x)`）保留高频振荡能力，适合拟合含大量干涉条纹的透射谱
- **mean(dim=2)** 把 1024→64 序列压缩为 256 维全局特征，交给全连接层生成 8002 个点
- 全连接 Dropout 0.2 防止对短序列（265 样本）过拟合

### 物理损失（`正向模型.py` 第 4 部分）

3 项子损失：

| 项 | 公式 | 含义 |
|---|------|------|
| ℒ_bc_s | `mean(ReLU(S))` | **非对称**：dB 透射值物理上必须 ≤ 0；只惩罚 S>0 |
| ℒ_pde | `0.5·(mean(ReLU(\|ΔS\|-0.1)) + mean(ReLU(\|ΔP\|-0.1)))` | S + P 两通道相邻点差值 ≤ 0.1 dB（光谱平滑性） |
| ℒ_bc_p | `mean(ReLU(\|P\|-1))` | 透射响应2 归一化后应在 [-1, 1] |

总物理损失：`ℒ_phy = ℒ_bc_s + ℒ_pde + ℒ_bc_p`
训练总损失：`ℒ_total = ℒ_data + λ_phy · ℒ_phy`

> ⚠️ ℒ_bc_s 是**非对称**约束（不是 `ReLU(|S|-1)`）。dB 透射值天然 ≤ 0，对负方向没有硬约束 —— 若改用对称形式，会在 S<-1 处（深谷）反向推离真实分布，导致训练失败。

## 训练超参

| 参数 | 正向 | 逆向 |
|------|------|------|
| 优化器 | Adam | Adam |
| 初始学习率 | 1e-4 | 1e-4 |
| weight_decay | 1e-5 | 1e-5 |
| 学习率调度 | ReduceLROnPlateau (factor=0.5, patience=10) | CosineAnnealingLR (T_max=600) |
| 最小学习率 | 1e-7 | 1e-6 |
| 批大小 | 16 | 16 |
| 早停耐心值 | 30 轮 | — |
| 最小训练轮数 | 100 | — |
| 收敛阈值 | 验证损失连续 5 轮变化 < 1e-6 | — |
| 数据划分 | 70% / 15% / 15% (seed=42) | 同 |

数据集规模：379 个样本，训练集 ≈ 265 / 验证集 ≈ 57 / 测试集 ≈ 57。

## 训练结果（参考）

正向模型在标准超参下（lr=1e-4, λ_phy=0, 150 轮）：

| 指标 | 数值 |
|---|---|
| 最终测试集 MSE | ~0.003 |
| 最佳验证 MSE | ~0.0028 |
| Loss 收敛所需轮数 | ~100 |

## 消融实验

本项目对 PINN 物理约束权重 `λ_phy` 进行了消融研究。

### 实验设计

- **前向消融**：固定 7 个权重 `0, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1`，观察物理约束强度对正向预测精度的影响
- **逆向消融（小跨度）**：固定 5 个权重 `0, 1e-5, 1e-4, 1e-3, 0.01`
- **逆向消融（大跨度）**：扩大范围到 `0, 1, 10, 1e-5`，观察极端权重下的退化

### 复现方法

编辑脚本顶部的 `lambda_phys` 变量（运行时会再次提示输入），重新训练即可。**注意**：物理约束是 "辅助信号" 而非主目标，纯数据（`λ_phy=0`）通常已是强基线，过大的 `λ_phy` 反而会损害数据拟合。

## 模块解耦

- `正向模型.py` — 完全自包含，不依赖任何 `物理模型/` 子目录
- `逆向模型.py` — 仅依赖 `正向模型.py` 中的 `ForwardNet`（用于物理约束中由参数反算光谱）
- 早期版本的 `物理模型/` 目录已移除

## 引用 / 致谢

- PINN 思路基于 Raissi 等人 [Physics-informed neural networks](https://doi.org/10.1016/j.jcp.2018.10.045)
- TFBG 物理背景：耦合模理论（Coupled-Mode Theory）

## 许可

仅供学术研究使用。
