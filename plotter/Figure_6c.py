import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import scienceplots
# plt.style.use(['science','nature'])
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
plt.style.use('own_style_1.mplstyle')
plt.rc('font', family='Times New Roman')

###############################################
#### 改变这两个参数，可以绘制不同的数据集的箱线图  ####
#### Change these two parameters to plot the boxplot of different datasets  ####
###############################################
data = 'HUST'
batch = 0
train_battery_num = [1,2,3,4]
total_df = []
for i in train_battery_num:
    df1 = pd.read_excel(f'../results_small-sample/processed_results/VerhulstPIKAN_opt/VerhulstPIKAN_opt_{data}_results_small_sample_{i}_best.xlsx',
                        engine='openpyxl',
                        nrows=10,
                        sheet_name=f'battery_mean_{batch}')
    df1['model'] = ['VerhulstPIKAN'] * 10
    df1['train num'] = [i] * 10
    df2 = pd.read_excel(f'../results_small-sample/processed_results/PIKAN_opt/PIKAN_opt_{data}_results_small_sample_{i}_best.xlsx',
                        engine='openpyxl',
                        nrows=10,
                        sheet_name=f'battery_mean_{batch}')
    df2['model'] = ['PIKAN'] * 10
    df2['train num'] = [i] * 10
    df3 = pd.read_excel(f'../results_small-sample/processed_results/VerhulstPINN_opt/VerhulstPINN_opt_{data}_results_small_sample_{i}_best.xlsx',
                        engine='openpyxl',
                        nrows=10,
                        sheet_name=f'battery_mean_{batch}')
    df3['model'] = ['VerhulstPINN'] * 10
    df3['train num'] = [i] * 10
    df4 = pd.read_excel(f'../results_small-sample/processed_results/PINN_opt/PINN_opt_{data}_results_small_sample_{i}_best.xlsx',
                        engine='openpyxl',
                        nrows=10,
                        sheet_name=f'battery_mean_{batch}')
    df4['model'] = ['PINN'] * 10
    df4['train num'] = [i] * 10
    df = pd.concat([df1, df2, df3, df4])
    total_df.append(df)
total_df = pd.concat(total_df)
total_df['RMSE'] = total_df['RMSE']*100
print(total_df)

if data in ['MIT', 'HUST']:
    title = data + ' dataset'
else:
    title = data + f' batch {batch+1}'

merge_df_keys = ['VerhulstPIKAN', 'PIKAN', 'VerhulstPINN', 'PINN']
colors = ["#fc9699","#ffaf7a","#73e4c2","#93d9fc"]
# 计算均值和标准差 (calculate mean and standard deviation)
mean_values = total_df.groupby(['model', 'train num'])['RMSE'].mean().reset_index()
std_values = total_df.groupby(['model', 'train num'])['RMSE'].std().reset_index()

fig, ax = plt.subplots(figsize=(8, 6),dpi=300)
sns.boxplot(x='model',y='RMSE',hue='train num',data=total_df,
               dodge=True,
               saturation=1,
               palette=colors,
               linewidth=1,
               ax=ax)

plt.xlabel(None)
# 坐标保留两位小数 (keep two decimal places of the coordinates)
# ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
plt.ylim([0.75,6.25])
plt.yticks([1.00,2.00,3.00,4.00,5.00,6.00],['1.00','2.00','3.00','4.00','5.00','6.00'])
plt.ylabel('RMSE (%)')
plt.grid(True)
plt.title('HUST dataset')

boxs = []
legends = []
for i in train_battery_num:
    boxs.append(plt.Rectangle((0, 0), 1, 1, fc=colors[i-1]))
    legends.append(f'{i} battery')
plt.legend(boxs, legends,
           loc='upper left',fontsize=18
           )
plt.tight_layout()
plt.savefig(f'../Figures/verhulst_small_sample_{data}.svg',format='svg')
plt.show()