'''
绘制小提琴图,并且把所有的数据集都绘制在一张图上
在4个数据集上的常规实验，绘制指标[MAE,MAPE,RMSE]的小提琴图，
在同一幅图上并对比Ours，MLP和CNN

English:
Draw a violin plot and plot all datasets on one figure
Common experiments on 4 data sets, plotting violin plots of indicators [MAE, MAPE, RMSE],
and comparing Ours, MLP, and CNN on the same figure
'''
import matplotlib.pyplot as plt
import scienceplots
# plt.style.use(['science','nature'])
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mtick
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
plt.style.use('own_style_3.mplstyle')
plt.rc('font', family='Times New Roman')

fig, axs = plt.subplots(4,3,figsize=(12,10),dpi=300)
# colors = ['#b8dff2','#abeadb',"#ffaf7a",'#ffb2b4']
colors = ["#fc9699","#ffaf7a","#73e4c2","#93d9fc"]
count = 0
for data in ['XJTU','TJU','MIT','HUST']:
    if data == 'XJTU':
        batches = [0,1,2,3,4,5]
    elif data == 'TJU':
        batches = [0,1,2]
    else:
        batches = [0]
    for batch in batches:
        ############################
        df_list = []
        for model in ['PIKAN', 'KAN', 'PINN', 'MLP']:
            if model == 'MLP' or model == 'KAN':
                df1 = pd.read_excel(f'../results_soh-estimation/processed_results/{model}/{model}_{data}_results.xlsx',
                                    engine='openpyxl',
                                    sheet_name=f'battery_mean_{batch}')
            else:
                df1 = pd.read_excel(f'../results_soh-estimation/processed_results/{model}_opt/{model}_opt_{data}_results_best.xlsx',
                                    engine='openpyxl',
                                    sheet_name=f'battery_mean_{batch}')

            df1['model'] = [model] * df1.shape[0]
            melted_df1 = pd.melt(df1, id_vars=['model'],
                                 value_vars=['MAE','MAPE','RMSE'],
                                 var_name='metric', value_name='error')
            df_list.append(melted_df1)
        if data in ['MIT', 'HUST']:
            title = data + ' dataset'
        else:
            title = data + f' batch {batch+1}'
        merge_df_keys = ['PIKAN', 'KAN', 'PINN', 'MLP']

        # 把三个DataFrame拼接起来
        # Concatenate three DataFrames
        df = pd.concat(df_list, axis=0)
        df = df.reset_index()
        df.drop('index', axis=1, inplace=True)
        df['metric'] = df['metric'].astype('category').cat.codes


        # 绘制小提琴图
        # Draw a violin plot
        col = count % 3
        row = count // 3
        print(data, batch, row, col)
        ax = axs[row, col]
        sns.violinplot(x='metric',y='error',hue='model',data=df,
                       density_norm='count',
                    #    scale='count',
                       inner='point',
                       dodge=True,
                       saturation=1,
                       palette=colors,
                       linewidth=1,  # 调整包络线粗细
                       color='gray',   # 设置包络线颜色为灰色
                       ax=ax)
        
        # # 恢复小提琴内部填充色
        # for i, violin in enumerate(ax.collections):
        #     original_color = colors[i % len(colors)]
        #     violin.set_facecolor(original_color)
        #     violin.set_edgecolor('gray')
        #     violin.set_alpha(0.8)        

        # 在绘制小提琴图后，添加以下代码来计算并绘制均值线和均值加减标准差线
        # After drawing the violin plot, add the following code to calculate and draw the mean line and the mean plus or minus the standard deviation line
        for i, metric in enumerate(['MAE', 'MAPE', 'RMSE']):
            for model in ['PIKAN', 'KAN', 'PINN', 'MLP']:
                model_mean = df[(df['model'] == model) & (df['metric'] == i)]['error'].mean()  # 计算每个模型的均值 (mean)
                model_std = df[(df['model'] == model) & (df['metric'] == i)]['error'].std()  # 计算每个模型的标准差 (standard deviation)
                # 计算均值线和标准差线的横坐标位置 (x position of the standard deviation line and mean line )
                offset = 0.27
                x_pos = i + (model == 'MLP') * 1.1* offset + (model == 'PINN') * 0.35* offset - (model == 'KAN') * 0.35* offset - (model == 'PIKAN') * 1.1* offset
                # 绘制标准差线 (draw the standard deviation line)
                ax.plot([x_pos, x_pos], [model_mean - model_std, model_mean + model_std], color='black', linestyle='-',
                        linewidth=1.2)
                # 绘制均值线 (draw the mean line)
                ax.plot([x_pos - 0.075, x_pos + 0.075], [model_mean, model_mean], color='red', linestyle='-', linewidth=1.2)

        # 设置x轴范围和标签位置 (set the x-axis range and label position)
        ax.set_xticklabels(['MAE', 'MAPE', 'RMSE'])
        ax.set_xlabel(None)
        ax.tick_params(axis='x', which='major', length=3, width=1, color='black')  # 主刻度线样式
        ax.tick_params(axis='x', which='minor', length=0)  # 移除副刻度线

        # 设置y轴范围 (set the y-axis range)

        # ax.set_ylim(0, 0.055)

        # 在y轴顶上加上百分号 (add a percentage sign on the top of the y-axis)
        def percentage(x, pos):
            if x >= 0.2:
                return '{:.0f}'.format(x * 100)
            else:
                return '{:.1f}'.format(x * 100)


        ax.yaxis.set_major_formatter(mtick.FuncFormatter(percentage))
        # 在y轴顶端添加百分号文本标签 (add a percentage text at the top of the y-axis)
        x_min, x_max = ax.get_xlim()
        y_max = ax.get_ylim()[1]
        ax.annotate('(%)', xy=(x_min, y_max), xytext=(-2, 3),
                    textcoords='offset points', ha='center', fontsize=14)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

        # 添加标题和标签 (add title and label)
        ax.set_title(title)
        ax.set_ylabel("Error")
        
        # 关闭图例 (remove the legend)
        ax.get_legend().remove()

        if count >= 11:
            break
        count += 1

    if count >= 11:
        break
# 添加图例 (add legend)
#axs[2, 3].set_visible(False)
# 关闭axs[2,3]的坐标轴 (remove the axis of axs[2,3])
axs[3, 2].axis('off')
boxs = []
for c in colors:
    box = plt.Rectangle((0, 0), 1, 1, fc=c)
    boxs.append(box)
mean_line = Line2D([0], [0], color='red', linestyle='-', linewidth=2)
std_line = Line2D([0, 0], [0, 1], color='black', linestyle='-', linewidth=2)
boxs.append(mean_line)
boxs.append(std_line)
legend_labels = ['PIKAN', 'KAN', 'PINN', 'MLP', 'Mean', 'Mean $\pm$ Std']
axs[3, 2].legend(
    handles=boxs,
    labels=legend_labels,
    loc=[0.15, -0.05],
    handlelength=4,     # 缩短句柄长度
    handleheight=1.5,   # 调整句柄高度
    fontsize=14,        # 调整字体大小
    ncol=1,             # 分为两列
    columnspacing=2.0,  # 列间距
    handletextpad=1.0,  # 句柄与文本的间距
    frameon=True       # 不显示图例边框
)

plt.tight_layout()
plt.savefig('../Figures/soh_estimation_violin_error.svg',format='svg')
plt.show()
