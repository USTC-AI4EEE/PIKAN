import numpy as np
import matplotlib.pyplot as plt
plt.style.use('../plotter/own_style_5.mplstyle') # 1
plt.rc('font', family='Times New Roman')
 
# 数据
classes = ['XJTU batch 1', 'TJU batch 3', 'MIT', 'HUST']
methods = ['PIKAN* Model','PIKAN* Symbolic']
pikan_before = np.array([0.0035, 0.0069, 0.0078, 0.0083])*100
pikan_after = np.array([0.0042, 0.0072, 0.0095, 0.0087])*100
colors = [
"#75A2F7",
"#357AFA",
] 

x = np.arange(len(classes))
width = 0.4

pikan_before_x = x + width
pikan_after_x = x + 2 * width

fig = plt.figure(figsize=(8,6),dpi=300)
# 绘图
plt.bar(pikan_before_x,pikan_before,width=width, color=colors[0],label=methods[0])
plt.bar(pikan_after_x,pikan_after,width=width, color=colors[1],label=methods[1])
plt.xticks(x + width + 0.2, labels=classes)
plt.ylabel('MAPE (%)')
plt.ylim((0.10,1.10))
plt.yticks([0.20,0.40,0.60,0.80,1.00],['0.20','0.40','0.60','0.80','1.00'])
#显示柱状图的高度文本
for i in range(len(classes)):
    plt.text(pikan_before_x[i],pikan_before[i], f"{pikan_before[i]:.2f}",va="bottom",ha="center",fontsize=14)
    plt.text(pikan_after_x[i],pikan_after[i], f"{pikan_after[i]:.2f}",va="bottom",ha="center",fontsize=14)
 
#显示图例
plt.legend(loc="upper left",fontsize=16)
plt.tight_layout()
plt.savefig('../Figures/MAPE_metric_pikan_before_and_after_symbolic_bold.svg',format='svg')
plt.show()