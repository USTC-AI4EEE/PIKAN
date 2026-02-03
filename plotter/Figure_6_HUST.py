'''
绘制小提琴图, 绘制HUST dataset的子图
在HUST数据集上的常规实验，绘制指标[RMSE]的小提琴图，
在同一幅图上并对比PIKAN，KAN，PINN，MLP，CNN和Transformer

English:
Draw a violin plot for HUST dataset subfigure
Common experiments on HUST dataset, plotting violin plot of indicator [RMSE],
and comparing PIKAN, KAN, PINN, MLP, CNN, and Transformer on the same figure
'''
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mtick
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
plt.style.use('plotter/own_style_4.mplstyle')
plt.rc('font', family='Times New Roman')
import matplotlib.colors as mcolors


def plot():
    # 创建单个子图，画布大小设置为(8,6)
    fig, ax = plt.subplots(1, 1, figsize=(8, 7), dpi=300)
    palette1 = sns.color_palette('Set2', 9)     # palette1[5]
    palette2 = sns.color_palette('pastel', 9)   # palette2[8]
    palette3 = sns.color_palette('Set3', 9)     # palette3[1]
    colors = [palette2[7],"#f8a0a3","#fcba8f","#f8f3a7","#73e4c2","#93d9fc"]
    
    # 选择要绘制的数据集：HUST dataset
    data = 'HUST'
    batch = 0
    
    ############################
    df_list = []
    for model in ['PIKAN', 'KAN', 'PINN', 'MLP', 'CNN','Transformer']:
        if model == 'MLP' or model == 'KAN' or model == 'CNN' or model == 'Transformer':
            df1 = pd.read_excel(f'results_soh-estimation/processed_results/{model}/{model}_{data}_results.xlsx',
                                engine='openpyxl',
                                sheet_name=f'battery_mean_{batch}')
        else:
            df1 = pd.read_excel(f'results_soh-estimation/processed_results/{model}_opt/{model}_opt_{data}_results_best.xlsx',
                                engine='openpyxl',
                                sheet_name=f'battery_mean_{batch}')

        df1['model'] = [model] * df1.shape[0]
        df_list.append(df1)
    
    # 合并数据
    df = pd.concat(df_list, axis=0)
    
    # 设置标题
    title = data + ' dataset'
    
    merge_df_keys = ['PIKAN', 'KAN', 'PINN', 'MLP','CNN','Transformer']
    
    # 绘制小提琴图，将模型作为x轴，RMSE作为y轴
    sns.violinplot(x='model', y='RMSE', data=df,
                density_norm='count',
                inner='point',
                dodge=False,  # 设置为False确保小提琴居中
                saturation=1,
                palette=colors,
                linewidth=2,  # 增加线条宽度
                color='gray',
                ax=ax)
    
    # 在绘制小提琴图后，添加均值线和均值加减标准差线
    for i, model in enumerate(merge_df_keys):
        model_mean = df[df['model'] == model]['RMSE'].mean()  # 计算每个模型的均值
        model_std = df[df['model'] == model]['RMSE'].std()    # 计算每个模型的标准差
        
        # 均值线（红色水平线）
        ax.plot([i - 0.3, i + 0.3], [model_mean, model_mean], color='red', linestyle='-', linewidth=2)
        
        # 标准差线（黑色垂直线）
        ax.plot([i, i], [model_mean - model_std, model_mean + model_std], color='black', linestyle='-', linewidth=2)

    # 设置x轴刻度标签
    ax.set_xticklabels(['PIKAN', 'KAN', 'PINN', 'MLP', 'CNN', 'Transformer'])
    ax.set_xlabel(None)
    
    # # 旋转x轴标签以便更好地显示
    # plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=18)
    
    # 设置y轴刻度标签字体大小
    ax.tick_params(axis='y', which='major', labelsize=20)
    
    ax.tick_params(axis='x', which='major', length=5, width=1.5, color='black')  # 主刻度线样式
    ax.tick_params(axis='x', which='minor', length=0)  # 移除副刻度线

    # 在y轴上使用百分比格式化
    def percentage(x, pos):
        if x >= 0.2:
            return '{:.0f}'.format(x * 100)
        else:
            return '{:.1f}'.format(x * 100)

    ax.yaxis.set_major_formatter(mtick.FuncFormatter(percentage))
    # 在y轴顶端添加百分号文本标签
    x_min, x_max = ax.get_xlim()
    y_max = ax.get_ylim()[1]
    ax.annotate('(%)', xy=(x_min, y_max), xytext=(-2, 3),
                textcoords='offset points', ha='center', fontsize=22)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    # 添加标题和标签
    ax.set_title(title, fontsize=28)  
    ax.set_ylabel("RMSE", fontsize=24) 
    
    # 添加图例
    boxs = []
    for c in colors:
        box = plt.Rectangle((0, 0), 1, 1, fc=c)
        boxs.append(box)
    mean_line = Line2D([0], [0], color='red', linestyle='-', linewidth=2)
    std_line = Line2D([0, 0], [0, 1], color='black', linestyle='-', linewidth=2)
    boxs.append(mean_line)
    boxs.append(std_line)

    plt.tight_layout(rect=[0, 0.1, 1, 1])  # 调整布局，为底部图例留出空间
    plt.savefig('Figures/soh_estimation_violin_error_hust.svg', format='svg', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    plot()