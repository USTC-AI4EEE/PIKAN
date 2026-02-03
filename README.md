# 基于物理信息柯尔莫哥洛夫-阿诺德网络(PIKAN)的SOH估计

PIKAN是一个先进的电池健康状态(SOH)估计框架，它创新性地结合了物理信息神经网络(PINNs)和柯尔莫哥洛夫-阿诺德网络(KANs)的优势，在多个电池数据集上实现了精确且可解释的SOH预测。

## 目录

- [项目概述](#项目概述)
- [主要特性](#主要特性)
- [技术栈](#技术栈)
- [安装](#安装)
- [数据集](#数据集)
- [使用方法](#使用方法)
- [实验结果](#实验结果)
- [可视化](#可视化)
- [贡献](#贡献)
- [许可证](#许可证)

## 项目概述

电池健康状态(SOH)估计是电池管理系统(BMS)中的关键任务。PIKAN引入了一种新颖的方法，该方法：

- 将物理信息约束与数据驱动学习相结合
- 利用柯尔莫哥洛夫-阿诺德网络增强函数逼近能力
- 通过符号回归提供可解释的数学表达式
- 在多个电池数据集上实现了最先进的性能

## 主要特性

- **物理信息学习**：将电池物理定律整合到神经网络训练过程中
- **柯尔莫哥洛夫-阿诺德网络**：利用KANs提高函数逼近能力
- **多数据集支持**：适用于HUST、MIT、TJU和XJTU等多种电池数据集
- **符号回归**：从训练模型中提取可解释的数学表达式
- **全面的可视化**：生成详细的图表用于结果分析
- **高效优化算法**：集成多种优化器以提升训练效率

## 技术栈

- Python 3.8+
- PyTorch（深度学习框架）
- NumPy（数值计算）
- Matplotlib & Seaborn（数据可视化）
- Pandas（数据处理）
- SciPy（科学计算）
- scikit-learn（机器学习库）

## 安装

1. 克隆仓库：
   ```bash
   git clone https://gitee.com/huangjh168/pikan.git
   cd PIKAN
   ```

2. 安装所需依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 确保安装了PyTorch并启用了适当的CUDA支持（如果使用GPU加速）。

## 数据集

本项目使用了多个公开的电池数据集来验证模型性能：

1. HUST电池数据集
- 包含多种电池类型和工况的数据
- 提供电压、电流、温度等多维特征
- 适用于动态工况下的SOH估计

2. MIT电池数据集
- 长期老化测试数据
- 涵盖多种充放电协议
- 用于评估模型的长期预测能力

3. TJU电池数据集
- 不同温度条件下的电池数据
- 适合研究温度对电池老化的影响
- 包含批量电池的测试结果

4. XJTU电池数据集
- 高精度测量数据
- 详细的电池退化过程记录
- 适用于高精度SOH估计研究

## 使用方法

### 模型训练
1. 准备数据：确保数据文件位于`Data/`目录下
2. 配置参数：修改配置文件以调整模型超参数
3. 运行训练脚本：(以XJTU数据集为例)
    ```bash
    # 使用超参数搜索
    python main_our_models_optimize_XJTU.py
    # 未使用超参数搜索
    python main_comparision_XJTU.py
    ```

### 结果分析
- 运行分析脚本以生成性能指标：
    ```bash
    python results_analysis_code/Model_optimize_XJTU_results.py
    python results_analysis_code/Model_optimize_XJTU_results_group_mean.py
    python results_analysis_code/Comparision_results_XJTU.py
    ```
- 查看生成的Excel结果文件和可视化图表

### 可视化
- 使用预定义的可视化脚本来生成图表：
    ```bash
    python plotter/Figure_4b_m6.py
    ```

## 实验结果

PIKAN在多个电池数据集上表现出色：

- **RMSE（均方根误差）**：0.5% - 1.5%
- **MAPE（平均绝对百分比误差）**：0.3% - 1.0%

## 可视化

本项目提供了丰富的可视化功能：

- SOH预测结果对比图
- 模型训练过程曲线
- 不同算法的性能比较
- 误差分布直方图
- 特征重要性分析图

所有可视化结果保存在`Figures/`目录下，支持多种格式（SVG、PNG等）。

## 贡献

欢迎社区贡献！如果您想为本项目做出贡献，请遵循以下步骤：

1. Fork 仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件了解详情。

## 致谢

感谢所有为本项目做出贡献的研究人员和开发者。特别感谢相关电池数据集的提供者以及PyTorch、KAN等开源项目的开发者。
        