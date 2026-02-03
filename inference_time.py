import torch
import numpy as np
import os
import time
from dataloader.dataloader import XJTUdata
from Model.Compare_Models import MLP, CNN, kan, Transformer, KAN_medium, KAN_small
from Model.PINN_opt import PINN_opt
from Model.PIKAN_opt import PIKAN_opt
from Model.PIKAN_hgs import PIKAN_hgs
from Model.PIKAN_small import PIKAN_small
import argparse
from utils.util import set_seed


# 获取参数
def get_args():
    parser = argparse.ArgumentParser('测试模型推理时间')
    parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--batch_size', type=int, default=1, help='推理时的batch size')
    parser.add_argument('--num_samples', type=int, default=1000, help='测试的样本数量')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    parser.add_argument('--results_dir', type=str, default='results_soh-estimation', help='模型参数保存目录')
    parser.add_argument('--batch', type=str, default='2C', help='XJTU数据集的批次')
    parser.add_argument('--test_seed', type=int, default=2025, help='测试时的随机种子')
    parser.add_argument('--warmup_runs', type=int, default=5, help='预热运行次数')
    
    parser.add_argument('--epochs', type=int, default=1, help='epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=5e-4, help='warmup lr')
    parser.add_argument('--final_lr', type=float, default=1e-4, help='final lr')
    parser.add_argument('--lr_F', type=float, default=1e-3, help='learning rate of F')
    parser.add_argument('--iter_per_epoch', type=int, default=1, help='iter per epoch')
    # 模型相关参数
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')
    parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')
    parser.add_argument('--alpha', type=float, default=1, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta', type=float, default=1, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    
    # 日志相关
    parser.add_argument('--save_folder', type=str, default=None, help='保存文件夹')
    parser.add_argument('--log_dir', type=str, default=None, help='日志目录')
    
    return parser.parse_args()


# 加载XJTU测试数据
def load_test_XJTU_data(args,model_name):
    if model_name == 'PIKAN_small':
        xjtu_batch_0 = torch.load('Data/XJTU_Data_Var2_batch_0.pt')
        testloader = xjtu_batch_0['test_loader']
    else:
        root = 'Data/XJTU data'
        data = XJTUdata(root=root, args=args)
        
        # 获取batch 1的测试文件
        test_list = []
        files = os.listdir(root)
        for file in files:
            if args.batch in file and ('4' in file or '8' in file):
                test_list.append(os.path.join(root, file))
        
        # 加载测试数据
        dataloader = data.read_all(specific_path_list=test_list)
        testloader = dataloader['test_3'] 
    return testloader  # 使用完整的测试集


# 准备测试样本
def prepare_test_samples(data_loader, num_samples, device):
    samples = []
    count = 0
    
    for x1, _, _, _ in data_loader:
        x1 = x1.to(device)
        for i in range(x1.shape[0]):
            samples.append(x1[i:i+1])  # 保持批次维度
            count += 1
            if count >= num_samples:
                return torch.cat(samples, dim=0)
    
    # 如果样本数量不足，返回所有可用样本
    return torch.cat(samples, dim=0)


# 加载模型
def load_model(model_name, args, device):
    if model_name == 'MLP':
        model = MLP().to(device)
        model_path = os.path.join(args.results_dir, 'MLP', f'XJTU-MLP results/0-0/Experiment1/model.pth')
    elif model_name == 'CNN':
        model = CNN().to(device)
        model_path = os.path.join(args.results_dir, 'CNN', f'XJTU-CNN results/0-0/Experiment1/model.pth')
    elif model_name == 'kan':
        model = kan().to(device)
        model_path = os.path.join(args.results_dir, 'KAN', f'XJTU-KAN results/0-0/Experiment1/model.pth')
    elif model_name == 'Transformer':
        model = Transformer().to(device)
        model_path = os.path.join(args.results_dir, 'Transformer', f'XJTU-Transformer results/0-0/Experiment1/model.pth')
    elif model_name == 'KAN_medium':
        model = KAN_medium().to(device)
        model_path = os.path.join(args.results_dir, 'KAN_medium', f'XJTU-KAN_medium results/0-0/Experiment1/model.pth')
    elif model_name == 'KAN_small':
        model = KAN_small().to(device)
        model_path = os.path.join(args.results_dir, 'KAN_small', f'XJTU-KAN_small results/0-0/Experiment1/model.pth')
    elif model_name == 'PINN_opt':
        model = PINN_opt(args).to(device)
        # model_path = os.path.join(args.results_dir, 'PINN_opt', f'XJTU results/0-0/Experiment1/best_results/best_model.pth')
        model_path = os.path.join(args.results_dir, 'PINN_opt', f'XJTU results/0-0/Experiment1/group25/model.pth')
    elif model_name == 'PIKAN_opt':
        model = PIKAN_opt(args).to(device)
        # model_path = os.path.join(args.results_dir, 'PIKAN_opt', f'XJTU results/0-0/Experiment1/best_results/best_model.pth')
        model_path = os.path.join(args.results_dir, 'PIKAN_opt', f'XJTU results/0-0/Experiment1/group25/model.pth')
    elif model_name == 'PIKAN_hgs':
        model = PIKAN_hgs(args).to(device)
        # model_path = os.path.join(args.results_dir, 'PIKAN_hgs', f'XJTU results/0-0/Experiment1/best_results/best_model.pth')
        model_path = os.path.join(args.results_dir, 'PIKAN_hgs', f'XJTU results/0-0/Experiment1/group25/model.pth')
    elif model_name == 'PIKAN_small':
        model = PIKAN_small(args).to(device)
        # model_path = os.path.join(args.results_dir, 'PIKAN_small', f'XJTU results/0-0/Experiment1/best_results/best_model.pth')
        model_path = os.path.join(args.results_dir, 'PIKAN_small', f'XJTU results/0-0/Experiment1/group25/model.pth')
    else:
        raise ValueError(f'未知模型: {model_name}')
    
    # 加载模型参数
    if os.path.exists(model_path):
        try:
            # 对于PINN类型的模型，需要加载两个部分的参数
            if model_name in ['PINN_opt', 'PIKAN_opt', 'PIKAN_hgs', 'PIKAN_small']:
                checkpoint = torch.load(model_path, map_location=device)
                model.solution_u.load_state_dict(checkpoint['solution_u'])
                model.dynamical_F.load_state_dict(checkpoint['dynamical_F'])
            else:
                # 对于比较模型，直接加载整个模型参数
                model.load_state_dict(torch.load(model_path, map_location=device))
            print(f'成功加载{model_name}模型参数: {model_path}')
        except Exception as e:
            print(f'加载{model_name}模型参数失败: {e}')
            print('使用随机初始化的模型进行测试')
    else:
        print(f'未找到{model_name}模型参数文件: {model_path}')
        print('使用随机初始化的模型进行测试')
    
    return model


# 测试模型推理时间
def test_inference_time(model, inputs, device, warmup_runs=5):
    model.eval()
    total_time = 0.0
    
    # 预热
    with torch.no_grad():
        for _ in range(warmup_runs):
            if hasattr(model, 'solution_u'):
                # 对于PINN类型模型，直接使用solution_u进行推理
                _ = model.solution_u(inputs[:10])
            else:
                _ = model(inputs[:10])
    
    if device == 'cuda':
        torch.cuda.synchronize()  # 确保CUDA操作完成
    
    # 实际测试
    with torch.no_grad():
        start_time = time.time()
        
        # 如果模型是PINN类型的，直接使用solution_u进行推理
        if hasattr(model, 'solution_u'):
            _ = model.solution_u(inputs)
        else:
            _ = model(inputs)
        
        if device == 'cuda':
            torch.cuda.synchronize()  # 确保CUDA操作完成
        
        end_time = time.time()
        total_time = end_time - start_time
    
    return total_time


# 主函数
def main():
    args = get_args()
    set_seed(args.test_seed)
    device = torch.device(args.device)
    print(f'使用设备: {device}')

    # 定义要测试的模型列表
    models_to_test = ['MLP', 'CNN', 'kan', 'KAN_medium', 'KAN_small', 'Transformer', 'PINN_opt', 'PIKAN_opt', 'PIKAN_hgs', 'PIKAN_small']
    
    # 存储每个模型的平均推理时间
    results = {}
    
    for model_name in models_to_test:
        print(f'\n测试{model_name}模型...')

        # 加载测试数据
        print('正在加载测试数据...')
        test_loader = load_test_XJTU_data(args,model_name)
        
        # 准备测试样本
        print(f'正在准备{args.num_samples}个测试样本...')
        test_samples = prepare_test_samples(test_loader, args.num_samples, device)
        actual_samples = test_samples.shape[0]
        print(f'成功准备{actual_samples}个测试样本')    

        # 加载模型
        model = load_model(model_name, args, device)
        
        # 测试推理时间
        total_time = test_inference_time(model, test_samples, device, args.warmup_runs)
        avg_time = total_time / actual_samples * 1000  # 转换为秒/1000个样本
        
        results[model_name] = avg_time
        print(f'{model_name}模型 - 总推理时间: {total_time:.6f}秒, 平均推理时间: {avg_time:.6f}秒/1000个样本')
    
    # 输出结果摘要
    print('\n===== 推理时间测试结果摘要 =====')
    for model_name, avg_time in sorted(results.items(), key=lambda x: x[1]):
        print(f'{model_name}: {avg_time:.6f}秒/1000个样本')
    
    # 保存结果到文件
    results_file = 'inference_time_results.txt'
    with open(results_file, 'w') as f:
        f.write('模型名称,平均推理时间(秒/1000个样本)\n')
        for model_name, avg_time in results.items():
            f.write(f'{model_name},{avg_time:.6f}\n')
    print(f'\n结果已保存到: {results_file}')


if __name__ == '__main__':
    main()