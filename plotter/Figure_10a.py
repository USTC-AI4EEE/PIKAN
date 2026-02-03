# -*- coding: utf-8 -*-
"""
参数敏感性分析热力图绘制
用于可视化不同超参数组合(α和β)下的模型性能(RMSE)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap

plt.rc('font', family='Times New Roman')
plt.style.use('plotter/own_style_1.mplstyle')
class ParameterSensitivityAnalysis:
    """参数敏感性分析类，用于绘制超参数α和β对模型性能的影响热力图"""
    
    def __init__(self, file_path, save_dir='Figures'):
        """
        初始化函数
        
        参数:
            file_path: 包含超参数和性能指标的表格文件路径
            save_dir: 图片保存目录
        """
        self.file_path = file_path
        self.save_dir = save_dir
        # 定义α和β的取值范围
        self.alpha_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        self.beta_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        
    def load_data(self, sheet_name=None):
        """加载并预处理数据"""
        # 读取表格文件的指定Sheet
        if sheet_name:
            df = pd.read_excel(self.file_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(self.file_path)
        
        # 确保group列和RMSE列存在
        if 'group' not in df.columns or 'RMSE' not in df.columns:
            raise ValueError(f"表格Sheet '{sheet_name}' 中缺少必要的列: 'group' 或 'RMSE'")
        
        # 检查数据是否包含49个group
        if len(df) != 49:
            print(f"警告: Sheet '{sheet_name}' 包含 {len(df)} 行，而不是期望的49行")
        
        # 按group列排序
        df = df.sort_values('group')
        
        return df
    
    def group_to_params(self, group_number):
        """
        将group编号转换为对应的α和β索引
        
        参数:
            group_number: group编号 (1-49)
        
        返回:
            (alpha_index, beta_index): α和β在取值列表中的索引
        """
        # 计算行和列索引 (从0开始)
        row = (group_number - 1) // 7
        col = (group_number - 1) % 7
        
        # 注意这里行列对应关系：
        # row对应alpha (从上到下)
        # col对应beta (从左到右)
        return row, col
    
    def prepare_heatmap_data(self, df):
        """准备热力图数据矩阵"""
        # 创建7x7的矩阵用于存储RMSE值
        rmse_matrix = np.zeros((7, 7))
        
        # 填充矩阵
        for _, row in df.iterrows():
            group_num = int(row['group'])
            rmse_val = row['RMSE']
            
            # 获取α和β的索引
            alpha_idx, beta_idx = self.group_to_params(group_num)
            
            # 由于热力图通常从上到下显示数据，这里可能需要反转行索引
            # 以确保alpha值从下到上递增
            rmse_matrix[6 - alpha_idx, beta_idx] = rmse_val
        
        return rmse_matrix
    
    def create_custom_colormap(self):
        """创建自定义颜色映射"""
        # 创建一个从蓝色到红色的渐变
        colors = [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),  # 蓝色
                  (0.9, 0.9, 0.9),  # 白色
                  (0.6823529411764706, 0.12156862745098039, 0.12156862745098039)]  # 红色
        
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
        return cmap
    
    def plot_heatmap(self, rmse_matrix, figsize=(10, 8)):
        """绘制热力图"""
        plt.figure(figsize=figsize)
        
        # 创建自定义颜色映射
        cmap = self.create_custom_colormap()
        
        # 绘制热力图 - 添加annot_kws参数来设置字体大小
        heatmap = sns.heatmap(
            rmse_matrix,
            annot=True,  # 显示数值
            fmt=".4f",  # 数值格式
            cmap=cmap,
            xticklabels=self.beta_values,
            yticklabels=self.alpha_values[::-1],  # 反转顺序，使最大的值在顶部
            cbar_kws={
                'label': 'RMSE',
                'pad': 0.025  # 调整颜色棒与热力图之间的间距（值越大间距越大）
            },
            annot_kws={'size': 16}  # 设置注释文本的字体大小为12
        )
        
        # 设置标题和标签 - 修改为Unicode字符形式
        plt.xlabel('β', fontsize=24)
        plt.ylabel('α', fontsize=24)
        
        # 设置刻度字体大小
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        
        # 优化布局
        plt.tight_layout()
        
        return plt.gcf()
    
    def save_figure(self, fig, filename='XJTU_PSA.svg'):
        """保存图片"""
        # 确保保存目录存在
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # 保存图片
        save_path = os.path.join(self.save_dir, filename)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")
        
    def run_single_sheet(self, sheet_name, filename=None):
        """处理单个Sheet并生成热力图"""
        try:
            # 加载数据
            df = self.load_data(sheet_name)
            print(f"成功加载Sheet '{sheet_name}' 数据，共{len(df)}条记录")
            
            # 准备热力图数据
            rmse_matrix = self.prepare_heatmap_data(df)
            print(f"Sheet '{sheet_name}' 热力图数据准备完成")
            
            # 绘制热力图
            fig = self.plot_heatmap(rmse_matrix)
            print(f"Sheet '{sheet_name}' 热力图绘制完成")
            
            # 如果没有提供文件名，则根据Sheet名称生成
            if filename is None:
                filename = f'XJTU_PSA_{sheet_name[-1]}_small_sample_1.svg' # 修改保存图片名称
                
            # 保存图片
            self.save_figure(fig, filename)
            
            return fig
        except Exception as e:
            print(f"处理Sheet '{sheet_name}' 过程中出错: {str(e)}")
            raise
    
    def run_all_sheets(self, sheet_names=None):
        """处理所有Sheet并为每个Sheet生成热力图"""
        if sheet_names is None:
            # 读取Excel文件中的所有Sheet名称
            excel_file = pd.ExcelFile(self.file_path)
            sheet_names = excel_file.sheet_names
            
        # 存储所有生成的图形
        figures = {}
        
        # 遍历所有Sheet
        for sheet_name in sheet_names:
            try:
                fig = self.run_single_sheet(sheet_name)
                figures[sheet_name] = fig
                # 关闭当前图形，避免内存占用过高
                plt.close(fig)
            except Exception as e:
                print(f"处理Sheet '{sheet_name}' 失败，跳过此Sheet")
                continue
                
        return figures

if __name__ == '__main__':
    # 设置表格文件路径
    file_path = 'results_soh-estimation/processed_results/PIKAN_opt/PIKAN_opt_XJTU_results_group_mean.xlsx'
    # file_path = 'results_small-sample/processed_results/PIKAN_opt/PIKAN_opt_XJTU_results_small_sample_1_group_mean.xlsx'
    # 创建分析器实例
    analyzer = ParameterSensitivityAnalysis(file_path)
    
    # 处理所有Sheet (Sheet1到Sheet6)
    # 如果Sheet名称不是Sheet1-Sheet6，可以根据实际情况修改
    sheet_names = [f'group_mean_{i}' for i in range(0, 6)]
    
    try:
        # 执行多Sheet分析和绘图
        figures = analyzer.run_all_sheets(sheet_names)
        print(f"成功处理了 {len(figures)} 个Sheet")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        # 可以选择显示单个Sheet进行调试
        # analyzer.run_single_sheet('Sheet1')
        # plt.show()