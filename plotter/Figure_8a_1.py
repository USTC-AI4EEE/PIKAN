import numpy as np
import matplotlib.pyplot as plt
plt.style.use('../plotter/own_style_5.mplstyle') # 1
plt.rc('font', family='Times New Roman')
 
# 数据
classes = ['XJTU batch 1', 'TJU batch 3', 'MIT', 'HUST']
methods = ['KAN Model','KAN Symbolic','PIKAN Model','PIKAN Symbolic']
# kan_before = [0.0037, 0.0042, 0.0035, 0.0042]
# kan_after = [0.0075, 0.0083, 0.0069, 0.0072]
# pikan_before = [0.0080, 0.0084, 0.0078, 0.0095]
# pikan_after = [0.0085, 0.0085, 0.0083, 0.0087]

kan_before = np.array([0.0037, 0.0075, 0.0080, 0.0085])*100
kan_after = np.array([0.0042, 0.0083, 0.0084, 0.0085])*100
pikan_before = np.array([0.0035, 0.0069, 0.0078, 0.0083])*100
pikan_after = np.array([0.0042, 0.0072, 0.0095, 0.0087])*100
colors = [
"#75A2F7",
"#357AFA",
"#F89574",
"#FC6635",
] 

x = np.arange(len(classes))
width = 0.2
kan_before_x = x
kan_after_x = x + width
pikan_before_x = x + 2 * width
pikan_after_x = x + 3* width
fig = plt.figure(figsize=(8,6),dpi=300)
# 绘图
plt.bar(kan_before_x, kan_before, width=width, color=colors[0], label=methods[0])
plt.bar(kan_after_x,kan_after,width=width,color=colors[1],label=methods[1])
plt.bar(pikan_before_x,pikan_before,width=width, color=colors[2],label=methods[2])
plt.bar(pikan_after_x,pikan_after,width=width, color=colors[3],label=methods[3])
plt.xticks(x + width + 0.1, labels=classes)
plt.ylabel('MAPE (%)')
plt.ylim((0.10,1.10))
plt.yticks([0.20,0.40,0.60,0.80,1.00],['0.20','0.40','0.60','0.80','1.00'])
#显示柱状图的高度文本
for i in range(len(classes)):
    plt.text(kan_before_x[i],kan_before[i], f"{kan_before[i]:.2f}",va="bottom",ha="center",fontsize=12)
    plt.text(kan_after_x[i],kan_after[i], f"{kan_after[i]:.2f}",va="bottom",ha="center",fontsize=12)
    plt.text(pikan_before_x[i],pikan_before[i], f"{pikan_before[i]:.2f}",va="bottom",ha="center",fontsize=12)
    plt.text(pikan_after_x[i],pikan_after[i], f"{pikan_after[i]:.2f}",va="bottom",ha="center",fontsize=12)
 
#显示图例
plt.legend(loc="upper left",fontsize=14)
plt.tight_layout()
plt.savefig('../Figures/MAPE_metric_kan_vs_pikan_before_and_after_symbolic_bold.svg',format='svg')
plt.show()