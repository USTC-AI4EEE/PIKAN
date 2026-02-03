'''
在2个数据集上的小样本实验，绘制指标[RMSE]的小提琴图，
在同一幅图上对比Ours，MLP和CNN

English:
Small sample experiments on 2 datasets, plotting violin plots of the indicator [RMSE],
Compare Ours, MLP and CNN on the same figure

'''
import matplotlib.pyplot as plt
import scienceplots
# plt.style.use(['science','nature'])
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
plt.style.use('plotter/own_style_7.mplstyle')   # 1
plt.rc('font', family='Times New Roman')

def plot():
    ###############################################
    #### 改变这两个参数，可以绘制不同的数据集的箱线图  ####
    #### Change these two parameters to plot the boxplot of different datasets  ####
    ###############################################
    data = 'HUST'
    batch = 0
    train_battery_num = [1,2,3,4]
    total_df = []
    for i in train_battery_num:
        df1 = pd.read_excel(f'results_small-sample/processed_results/PIKAN_opt/PIKAN_opt_{data}_results_small_sample_{i}_best.xlsx',
                            engine='openpyxl',
                            nrows=10,
                            sheet_name=f'battery_mean_{batch}')
        df1['model'] = ['PIKAN'] * 10
        df1['train num'] = [i] * 10
        df2 = pd.read_excel(f'results_small-sample/processed_results/KAN/KAN_{data}_results_small_sample_{i}.xlsx',
                            engine='openpyxl',
                            nrows=10,
                            sheet_name=f'battery_mean_{batch}')
        df2['model'] = ['KAN'] * 10
        df2['train num'] = [i] * 10
        df3 = pd.read_excel(f'results_small-sample/processed_results/PINN_opt/PINN_opt_{data}_results_small_sample_{i}_best.xlsx',
                            engine='openpyxl',
                            nrows=10,
                            sheet_name=f'battery_mean_{batch}')
        df3['model'] = ['PINN'] * 10
        df3['train num'] = [i] * 10
        df4 = pd.read_excel(f'results_small-sample/processed_results/Transformer/Transformer_{data}_results_small_sample_{i}.xlsx',
                            engine='openpyxl',
                            nrows=10,
                            sheet_name=f'battery_mean_{batch}')
        df4['model'] = ['Transformer'] * 10
        df4['train num'] = [i] * 10
        df5 = pd.read_excel(f'results_small-sample/processed_results/MLP/MLP_{data}_results_small_sample_{i}.xlsx',
                            engine='openpyxl',
                            nrows=10,
                            sheet_name=f'battery_mean_{batch}')
        df5['model'] = ['MLP'] * 10
        df5['train num'] = [i] * 10
        df6 = pd.read_excel(f'results_small-sample/processed_results/CNN/CNN_{data}_results_small_sample_{i}.xlsx',
                            engine='openpyxl',
                            nrows=10,
                            sheet_name=f'battery_mean_{batch}')
        df6['model'] = ['CNN'] * 10
        df6['train num'] = [i] * 10
        # df6 = pd.read_excel(f'results_small-sample/processed_results/Transformer/Transformer_{data}_results_small_sample_{i}.xlsx',
        #                     engine='openpyxl',
        #                     nrows=10,
        #                     sheet_name=f'battery_mean_{batch}')
        # df6['model'] = ['Transformer'] * 10
        # df6['train num'] = [i] * 10
        df = pd.concat([df1, df2, df3, df4, df5, df6])
        total_df.append(df)
    total_df = pd.concat(total_df)
    print(total_df)

    if data in ['MIT', 'HUST']:
        title = data + ' dataset'
    else:
        title = data + f' batch {batch+1}'

    merge_df_keys = ['PIKAN', 'KAN', 'PINN','Transformer', 'MLP', 'CNN']
    colors = ["#e8f8ff","#8de6ff","#20d2ff","#00aeffff"]

    # 计算均值和标准差 (calculate mean and standard deviation)
    mean_values = total_df.groupby(['model', 'train num'])['RMSE'].mean().reset_index()
    std_values = total_df.groupby(['model', 'train num'])['RMSE'].std().reset_index()

    fig, ax = plt.subplots(figsize=(12,9),dpi=300) # 8, 6
    sns.violinplot(x='model',y='RMSE',hue='train num',data=total_df,
                scale='count',
                inner='point',
                dodge=True,
                saturation=1,
                palette=colors,
                linewidth=1,
                color='gray',
                bw_adjust=0.5,
                ax=ax)

    plt.xlabel(None)
    # 坐标保留两位小数 (keep two decimal places of the coordinates)
    # ax.yaxis.set_major_formatter(plt.FormatSTransformerormatter('%.2f'))
    # plt.ylim([-0.01,0.17])
    # plt.yticks([0.00,0.03,0.06,0.09,0.12,0.15],['0.00','0.03','0.06','0.09','0.12','0.15'])
    plt.title('HUST dataset')

    # 添加均值线和标准差线 (add mean line and standard deviation line)
    for model in merge_df_keys:
        for train_num in train_battery_num:
            mean = mean_values[(mean_values['model'] == model) & (mean_values['train num'] == train_num)]['RMSE'].values[0]
            std = std_values[(std_values['model'] == model) & (std_values['train num'] == train_num)]['RMSE'].values[0]

            # 根据模型和训练数量确定x坐标位置 (determine the x-coordinate position based on the model and the number of training)
            x_position = merge_df_keys.index(model) + (train_num - 1) * 0.2 - 0.3

            ax.plot(
                [x_position-0.075, x_position+0.075],  # x坐标位置 (the x-coordinate position)
                [mean,mean],  # y坐标位置（均值） (y-coordinate position (mean))
                linestyle='-',
                linewidth=1.2,
                color='red',
                label=f'mean ± std ({model}, {train_num})'
            )

            ax.plot(
                [x_position, x_position],  # x坐标位置（垂直线）(the x-coordinate position (vertical line))
                [mean - std, mean + std],  # y坐标位置（上下标准差线） (y-coordinate position (upper and lower standard deviation line))
                linestyle='-',
                linewidth=1.2,
                color='black'
            )


    boxs = []
    legends = []
    for i in train_battery_num:
        boxs.append(plt.Rectangle((0, 0), 1, 1, fc=colors[i-1]))
        legends.append(f'{i} battery')
    mean_line = Line2D([0], [0], color='red', linestyle='-', linewidth=3)
    std_line = Line2D([0,0], [0,1], color='black', linestyle='-', linewidth=3)
    boxs.append(mean_line)
    boxs.append(std_line)
    legends.append('mean')
    legends.append('mean ± std')
    plt.legend(boxs, legends,
            loc='upper left',fontsize=22
            )
    plt.tight_layout()
    plt.savefig(f'Figures/small_sample_HUST_m6.svg',format='svg')
    plt.show()

if __name__ == '__main__':
    plot()