'''
绘制小提琴图,并且把所有的数据集都绘制在一张图上
在4个数据集上的常规实验，绘制指标[RMSE]的小提琴图，
在同一幅图上并对比Ours，MLP和CNN

English:
Draw a violin plot and plot all datasets on one figure
Common experiments on 4 data sets, plotting violin plots of indicators [RMSE],
and comparing Ours, MLP, and CNN on the same figure
'''
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mtick
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
plt.style.use('plotter/own_style_3.mplstyle')
plt.rc('font', family='Times New Roman')
import matplotlib.colors as mcolors


def plot():
    # 更改布局为3行4列
    fig, axs = plt.subplots(4, 3, figsize=(20, 16), dpi=300)
    palette1 = sns.color_palette('Set2', 9)     # palette1[5]
    palette2 = sns.color_palette('pastel', 9)   # palette2[8]
    palette3 = sns.color_palette('Set3', 9)     # palette3[1]
    colors = [palette2[7],"#f8a0a3","#fcba8f","#f8f3a7","#73e4c2","#93d9fc"]
    
    # 收集所有数据集和批次信息
    all_datasets = []
    for data in ['XJTU','TJU','MIT','HUST']:
        if data == 'XJTU':
            batches = [0,1,2,3,4,5]
        elif data == 'TJU':
            batches = [0,1,2]
        else:
            batches = [0]
        
        for batch in batches:
            all_datasets.append((data, batch))
            if len(all_datasets) >= 11:  # 总共需要11张子图
                break
        if len(all_datasets) >= 11:
            break
    
    # 调整子图顺序，确保每行居中
    # 第一行：4张，第二行：4张，第三行：3张（居中显示）
    adjusted_datasets = []
    adjusted_datasets.extend(all_datasets[:4])  # 第一行4张
    adjusted_datasets.extend(all_datasets[4:8])  # 第二行4张
    adjusted_datasets.extend(all_datasets[8:11])  # 第三行3张
    
    for idx, (data, batch) in enumerate(adjusted_datasets):
        ############################
        df_list = []
        for model in ['PIKAN', 'KAN', 'PINN','Transformer', 'MLP', 'CNN']:
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
        
        if data in ['MIT', 'HUST']:
            title = data + ' dataset'
        else:
            title = data + f' batch {batch+1}'
        
        merge_df_keys = ['PIKAN', 'KAN', 'PINN','Transformer', 'MLP', 'CNN']

        # 计算行和列
        row = idx // 3
        col = idx % 3
        
        # 获取当前子图
        ax = axs[row, col]
        
        # 绘制小提琴图，将模型作为x轴，RMSE作为y轴
        # 增加linewidth使小提琴线条更粗，调整dodge=False确保小提琴居中
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

        # 设置x轴刻度标签，字体大小增加2
        ax.set_xticklabels(['PIKAN', 'KAN', 'PINN','Transformer', 'MLP', 'CNN'])
        ax.set_xlabel(None)
        
        # 旋转x轴标签以便更好地显示，字体大小增加2
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=18)  # 从12增加到14
        
        # 设置y轴刻度标签字体大小，增加2
        ax.tick_params(axis='y', which='major', labelsize=16)  # 增加字体大小
        
        ax.tick_params(axis='x', which='major', length=5, width=1.5, color='black')  # 主刻度线样式，增加长度和宽度
        ax.tick_params(axis='x', which='minor', length=0)  # 移除副刻度线

        # 在y轴上使用百分比格式化
        def percentage(x, pos):
            if x >= 0.2:
                return '{:.0f}'.format(x * 100)
            else:
                return '{:.1f}'.format(x * 100)

        ax.yaxis.set_major_formatter(mtick.FuncFormatter(percentage))
        # 在y轴顶端添加百分号文本标签，字体大小增加2
        x_min, x_max = ax.get_xlim()
        y_max = ax.get_ylim()[1]
        ax.annotate('(%)', xy=(x_min, y_max), xytext=(-2, 3),
                    textcoords='offset points', ha='center', fontsize=18)  # 从14增加到16
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

        # 添加标题和标签，字体大小增加2
        ax.set_title(title, fontsize=24)  
        ax.set_ylabel("RMSE", fontsize=20) 
        
        # 注释掉这行代码，因为violinplot没有添加图例，所以不需要移除
        # ax.get_legend().remove()
    
    # 隐藏第3行第2列的子图
    axs[3, 2].axis('off')
    
    # 设置子图间距
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    
    # 添加图例
    boxs = []
    for c in colors:
        box = plt.Rectangle((0, 0), 1, 1, fc=c)
        boxs.append(box)
    mean_line = Line2D([0], [0], color='red', linestyle='-', linewidth=2)
    std_line = Line2D([0, 0], [0, 1], color='black', linestyle='-', linewidth=2)
    boxs.append(mean_line)
    boxs.append(std_line)
    legend_labels = ['PIKAN', 'KAN', 'PINN','Transformer', 'MLP', 'CNN', 'Mean', 'Mean $\pm$ Std']
    
    # 在底部居中位置添加图例，字体大小增加2
    axs[3,2].legend(
        handles=boxs,
        labels=legend_labels,
        loc='lower center',
        bbox_to_anchor=(0.45, -0.05),
        handlelength=4,
        handleheight=1.5,
        fontsize=20,  # 从16增加到18
        ncol=2,
        columnspacing=1.0,
        handletextpad=1.0,
        frameon=True
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])  # 调整布局，为底部图例留出空间
    plt.savefig('Figures/soh_estimation_violin_error_m6.svg', format='svg', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    plot()