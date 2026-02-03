import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle  # 用于绘制边框
import numpy as np
plt.rc('font', family='Times New Roman')


def plot():
    # 模拟数据（替换为实际数据），按题目结构：4个数据集，每个数据集16个特征
    # 这里用随机数模拟，实际使用时请替换为真实相关系数矩阵
    xjtu_corr_matrix = np.load('Data/xjtu_corr_matrix.npy')
    tju_corr_matrix = np.load('Data/tju_corr_matrix.npy')
    mit_corr_matrix = np.load('Data/mit_corr_matrix.npy')
    hust_corr_matrix = np.load('Data/hust_corr_matrix.npy')
    data = {
        "XJTU": xjtu_corr_matrix,  # 6行（XJTU有6组行数据），16列特征
        "TJU": tju_corr_matrix,   # 3行（TJU有3组行数据），16列特征
        "MIT": mit_corr_matrix.reshape(1,-1),   # 1行（MIT有1组行数据），16列特征
        "HUST": hust_corr_matrix.reshape(1,-1)   # 1行（HUST有1组行数据），16列特征
    }

    # 数据集名称及对应行数，用于布局
    datasets = ["XJTU", "TJU", "MIT", "HUST"]
    row_counts = [6, 3, 1, 1]

    # 创建画布，根据数据集行数调整高度
    fig, axes = plt.subplots(
        nrows=4, ncols=1, 
        figsize=(8, 7.2),  # 可根据实际需求调整宽高
        gridspec_kw={
            "height_ratios": row_counts,
            # "hspace": 0.1  # 增加子图之间的间距
        }
    )

    # 颜色映射，可根据喜好换，比如 'coolwarm' 适合相关系数（-1到1）
    cmap = plt.get_cmap('coolwarm')

    for i, (ds_name, ax) in enumerate(zip(datasets, axes)):
        matrix = data[ds_name]

        # 绘制热力图
        im = ax.imshow(
            matrix, 
            cmap=cmap, 
            vmin=-1, 
            vmax=1, 
            aspect='equal'  # 使色块为正方形
        )
        
        # 手动添加白色边框
        for row in range(matrix.shape[0]):
            for col in range(16):
                # 添加白色边框
                rect = Rectangle(
                    (col - 0.5, row - 0.5),  # 位置
                    1, 1,                   # 宽度和高度
                    linewidth=0.05,          # 线宽
                    edgecolor='white',      # 边框颜色
                    facecolor='none'        # 填充颜色
                )
                ax.add_patch(rect)
        
        # 设置x轴刻度值（去除刻度线）
        ax.set_xticks(np.arange(16))
        ax.set_xticklabels(np.arange(1, 17))
        ax.tick_params(axis='x', which='both', length=0, labelsize=12)  # 去除x轴刻度线
        
        # # 设置y轴标签（去除刻度和刻度线）
        # ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels([])
        ax.tick_params(axis='y', which='both', length=0, labelleft=True)  # 去除y轴刻度线 
        
        # 标注数值（颜色改为白色）
        for row in range(matrix.shape[0]):
            for col in range(16):
                val = matrix[row, col]
                if val > -0.5 and val < 0.5:
                    ax.text(
                        col, row,
                        f"{val:.2f}",
                        ha='center', va='center',
                        color='black',       
                        fontsize=12,
                        # fontweight='bold'    # 加粗字体，提高可读性
                    )
                else:
                    ax.text(
                        col, row,
                        f"{val:.2f}",
                        ha='center', va='center',
                        color='white',       # 文本颜色设为白色
                        fontsize=12,
                        # fontweight='bold'    # 加粗字体，提高可读性
                    )

        # 将子图的边框（spines）设置为白色
        for spine in ax.spines.values():
            spine.set_color('white')
            spine.set_linewidth(0.1)  # 可调整边框粗细

        # # 设置子图标题
        # ax.set_title(ds_name, fontsize=10, y=1.02)

    # 调整整体布局，让子图紧凑
    plt.tight_layout()
    plt.subplots_adjust(top=1)
    # # 添加右侧颜色条
    # fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', shrink=0.6)
    # # 设置整个图的标题
    # fig.suptitle("Correlation Heatmap of 16 Health Features with SOH", fontsize=12, y=0.92)
    plt.show()
    plt.savefig('Figures/correlation_heatmap.svg',format='svg')


if __name__ == '__main__':
    plot()