import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('plotter/own_style_1.mplstyle')
plt.rc('font', family='Times New Roman')


def plot():
    # 假设的数据 - 实际使用时替换为你的真实数据
    data = {
        'Models': ['PIKAN* Model', 'PIKAN* Model', 'PIKAN* Symbolic', 'PIKAN* Symbolic', 'PIKAN* Model', 'PIKAN* Model', 'PIKAN* Symbolic', 'PIKAN* Symbolic'],
        'Datasets': ['XJTU batch 1', 'XJTU batch 1', 'XJTU batch 1', 'XJTU batch 1', 'TJU batch 3', 'TJU batch 3', 'TJU batch 3', 'TJU batch 3'],
        'Metrics': ['MAPE', 'RMSE', 'MAPE', 'RMSE', 'MAPE', 'RMSE', 'MAPE', 'RMSE'],
        'Values': [0.35, 0.40, 0.42, 0.50, 0.69, 0.58, 0.72, 0.62]  # 示例数值
    }

    df = pd.DataFrame(data)

    # 设置图像尺寸为8:6
    fig, ax = plt.subplots(figsize=(8, 6))

    # 定义柱状图的位置和宽度
    bar_width = 0.22
    index = np.arange(2)  # 两个数据集

    # 提取不同模型和指标的数据
    mape_a1 = df[(df['Models'] == 'PIKAN* Model') & (df['Datasets'] == 'XJTU batch 1') & (df['Metrics'] == 'MAPE')]['Values'].values[0]
    rmse_a1 = df[(df['Models'] == 'PIKAN* Model') & (df['Datasets'] == 'XJTU batch 1') & (df['Metrics'] == 'RMSE')]['Values'].values[0]
    mape_b1 = df[(df['Models'] == 'PIKAN* Symbolic') & (df['Datasets'] == 'XJTU batch 1') & (df['Metrics'] == 'MAPE')]['Values'].values[0]
    rmse_b1 = df[(df['Models'] == 'PIKAN* Symbolic') & (df['Datasets'] == 'XJTU batch 1') & (df['Metrics'] == 'RMSE')]['Values'].values[0]

    mape_a2 = df[(df['Models'] == 'PIKAN* Model') & (df['Datasets'] == 'TJU batch 3') & (df['Metrics'] == 'MAPE')]['Values'].values[0]
    rmse_a2 = df[(df['Models'] == 'PIKAN* Model') & (df['Datasets'] == 'TJU batch 3') & (df['Metrics'] == 'RMSE')]['Values'].values[0]
    mape_b2 = df[(df['Models'] == 'PIKAN* Symbolic') & (df['Datasets'] == 'TJU batch 3') & (df['Metrics'] == 'MAPE')]['Values'].values[0]
    rmse_b2 = df[(df['Models'] == 'PIKAN* Symbolic') & (df['Datasets'] == 'TJU batch 3') & (df['Metrics'] == 'RMSE')]['Values'].values[0]

    # 绘制柱状图
    bar1 = plt.bar(index - bar_width*1.5, [mape_a1, mape_a2], bar_width, label='PIKAN* Model MAPE', color="#75A2F7")
    bar2 = plt.bar(index - bar_width/2, [mape_b1, mape_b2], bar_width, label='PIKAN* Symbolic MAPE', color="#357AFA")
    bar3 = plt.bar(index + bar_width/2, [rmse_a1, rmse_a2], bar_width, label='PIKAN* Model RMSE', color="#F89574")
    bar4 = plt.bar(index + bar_width*1.5, [rmse_b1, rmse_b2], bar_width, label='PIKAN* Symbolic RMSE', color="#FC6635")


    plt.ylim((0.00,1.00))
    plt.yticks([0.10,0.30,0.50,0.70,0.90],['0.10','0.30','0.50','0.70','0.90'])
    # 添加标签和标题
    # plt.xlabel('Datasets')
    plt.ylabel('Metrics (%)')
    # plt.title('不同模型在各数据集上的MAPE和RMSE对比', fontsize=14)
    plt.xticks(index, ['XJTU batch 1', 'TJU batch 3'])
    plt.legend(loc="upper left",fontsize=14)

    # 在柱状图上添加数值标签
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom',fontsize=14)

    add_labels(bar1)
    add_labels(bar2)
    add_labels(bar3)
    add_labels(bar4)

    # 调整布局
    plt.tight_layout()
    plt.savefig('Figures/metrics_pikan_before_and_after_symbolic.svg',format='svg')
    # 显示图形
    plt.show()


if __name__ == '__main__':
    plot()
