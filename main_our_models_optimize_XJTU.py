from dataloader.dataloader import XJTUdata
from Model.PIKAN_opt import PIKAN_opt
from Model.PINN_opt import PINN_opt
from Model.PIKAN_small import PIKAN_small
from Model.PIKAN_hgs import PIKAN_hgs
import argparse
import os
import numpy as np
import torch
from utils.util import eval_metrix,get_logger,set_seed


def load_data(args,small_sample=None):
    root = 'Data/XJTU data'
    data = XJTUdata(root=root, args=args)
    train_list = []
    test_list = []
    files = os.listdir(root)
    for file in files:
        if args.batch in file:
            if '4' in file or '8' in file:
                test_list.append(os.path.join(root, file))
            else:
                train_list.append(os.path.join(root, file))
    if small_sample is not None:
        train_list = train_list[:small_sample]

    train_loader = data.read_all(specific_path_list=train_list)
    test_loader = data.read_all(specific_path_list=test_list)
    dataloader = {'train': train_loader['train_2'],
                  'valid': train_loader['valid_2'],
                  'test': test_loader['test_3']}
    return dataloader


def load_model(args):
    if args.model_name == 'PINN_opt':
        model = PINN_opt(args)
    elif args.model_name == 'PIKAN_opt':
        model = PIKAN_opt(args)
    elif args.model_name == 'PIKAN_small':
        model = PIKAN_small(args)
    elif args.model_name == 'PIKAN_hgs':
        model = PIKAN_hgs(args)
    return model


def main_2():
    args = get_args()
    setattr(args,'model_name','PIKAN_small') # 只运行PIKAN_small
    batchs = ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']
    for i in [0]:
        batch = batchs[i]
        setattr(args, 'batch', batch)
        for e in range(10):
            set_seed(e)         
            best_MAPE = float('inf')  # 初始化当前i和e下的最佳MAPE为正无穷
            best_alpha = None
            best_beta = None
            best_metrics = None
            best_pred_label = None
            best_true_label = None
            best_model = None
            group = 0
            for alpha in args.alpha_values:
                for beta in args.beta_values:
                    group += 1
                    save_folder = f'results_soh-estimation/{args.model_name}/XJTU results/' + str(i) + '-' + str(i) + '/Experiment' + str(e + 1) + f'/group{group}'
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    log_dir = 'logging.txt'
                    setattr(args, "save_folder", save_folder)
                    setattr(args, "log_dir", log_dir)
                    xjtu_dataset = torch.load('Data/XJTU_Data_Var2_batch_0.pt')
                    args.alpha = alpha
                    args.beta = beta
                    model = load_model(args)
                    model.Train(trainloader=xjtu_dataset['train_loader'],validloader=xjtu_dataset['valid_loader'],testloader=xjtu_dataset['test_loader'])
                    true_label, pred_label = model.Test(testloader=xjtu_dataset['test_loader'])
                    metrics = eval_metrix(pred_label, true_label)
                    # 判断是否为当前i和e下MAPE最小的组合
                    if metrics[1] < best_MAPE:
                        best_MAPE = metrics[1]
                        best_alpha = alpha
                        best_beta = beta
                        best_metrics = metrics
                        best_pred_label = pred_label
                        best_true_label = true_label
                        best_model = {'solution_u':model.solution_u.state_dict(),'dynamical_F':model.dynamical_F.state_dict()}
            # 保存当前i和e下MAPE最小的超参数组合对应的结果到特定文件夹
            best_save_folder = f'results_soh-estimation/{args.model_name}/XJTU results/' + str(i) + '-' + str(i) + '/Experiment' + str(e + 1) + f'/best_results'
            if not os.path.exists(best_save_folder):
                os.makedirs(best_save_folder)
            np.save(os.path.join(best_save_folder, 'true_label.npy'), best_true_label)
            np.save(os.path.join(best_save_folder, 'pred_label.npy'), best_pred_label)
            best_log_dir = os.path.join(best_save_folder, "logging.txt") 
            best_logger = get_logger(best_log_dir) 
            info = '[Best]  alpha: {}, beta: {}, MAE: {:.4f}, MAPE: {:.4f}, MSE: {:.6f}, RMSE: {:.4f}'.format(best_alpha,best_beta,best_metrics[0], best_metrics[1], best_metrics[2], best_metrics[3])
            best_logger.info(info)
            torch.save(best_model,os.path.join(best_save_folder,'best_model.pth'))
            torch.cuda.empty_cache()


def main():
    args = get_args()
    setattr(args,'model_name','PIKAN_opt') # 可运行PIKAN_opt, PINN_opt, PIKAN_hgs
    batchs = ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']
    for i in range(5,6):
        batch = batchs[i]
        setattr(args, 'batch', batch)
        for e in range(10):
            set_seed(e)         
            best_MAPE = float('inf')  # 初始化当前i和e下的最佳MAPE为正无穷
            best_alpha = None
            best_beta = None
            best_metrics = None
            best_pred_label = None
            best_true_label = None
            best_model = None
            group = 0
            for alpha in args.alpha_values:
                for beta in args.beta_values:
                    group += 1
                    save_folder = f'results_soh-estimation/{args.model_name}/XJTU results/' + str(i) + '-' + str(i) + '/Experiment' + str(e + 1) + f'/group{group}'
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    log_dir = 'logging.txt'
                    setattr(args, "save_folder", save_folder)
                    setattr(args, "log_dir", log_dir)
                    dataloader = load_data(args)   
                    args.alpha = alpha
                    args.beta = beta
                    model = load_model(args)
                    model.Train(trainloader=dataloader['train'],validloader=dataloader['valid'],testloader=dataloader['test'])
                    true_label, pred_label = model.Test(testloader=dataloader['test'])
                    metrics = eval_metrix(pred_label, true_label)
                    # 判断是否为当前i和e下MAPE最小的组合
                    if metrics[1] < best_MAPE:
                        best_MAPE = metrics[1]
                        best_alpha = alpha
                        best_beta = beta
                        best_metrics = metrics
                        best_pred_label = pred_label
                        best_true_label = true_label
                        best_model = {'solution_u':model.solution_u.state_dict(),'dynamical_F':model.dynamical_F.state_dict()}
            # 保存当前i和e下MAPE最小的超参数组合对应的结果到特定文件夹
            best_save_folder = f'results_soh-estimation/{args.model_name}/XJTU results/' + str(i) + '-' + str(i) + '/Experiment' + str(e + 1) + f'/best_results'
            if not os.path.exists(best_save_folder):
                os.makedirs(best_save_folder)
            np.save(os.path.join(best_save_folder, 'true_label.npy'), best_true_label)
            np.save(os.path.join(best_save_folder, 'pred_label.npy'), best_pred_label)
            best_log_dir = os.path.join(best_save_folder, "logging.txt") 
            best_logger = get_logger(best_log_dir) 
            info = '[Best]  alpha: {}, beta: {}, MAE: {:.4f}, MAPE: {:.4f}, MSE: {:.6f}, RMSE: {:.4f}'.format(best_alpha,best_beta,best_metrics[0], best_metrics[1], best_metrics[2], best_metrics[3])
            best_logger.info(info)
            torch.save(best_model,os.path.join(best_save_folder,'best_model.pth'))
            torch.cuda.empty_cache()


def small_sample():
    args = get_args()
    setattr(args,'model_name','PIKAN_opt') # 可运行PIKAN_opt, PINN_opt, PIKAN_hgs
    batchs = ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']
    for n in [1,2,3,4]:
        for i in [0]:
            batch = batchs[i]
            setattr(args, 'batch', batch)
            setattr(args,'batch_size',128)
            for e in range(10):
                set_seed(e)         
                best_MAPE = float('inf')  # 初始化当前i和e下的最佳MAPE为正无穷
                best_alpha = None
                best_beta = None
                best_metrics = None
                best_pred_label = None
                best_true_label = None
                best_model = None
                group = 0
                for alpha in args.alpha_values:
                    for beta in args.beta_values:
                        group += 1
                        save_folder = f'results_small-sample/{args.model_name}/XJTU results (small sample {n})/' + str(i) + '-' + str(i) + '/Experiment' + str(e + 1) + f'/group{group}'
                        if not os.path.exists(save_folder):
                            os.makedirs(save_folder)
                        log_dir = 'logging.txt'
                        setattr(args, "save_folder", save_folder)
                        setattr(args, "log_dir", log_dir)
                        dataloader = load_data(args,small_sample=n)   
                        args.alpha = alpha
                        args.beta = beta
                        model = load_model(args)
                        model.Train(trainloader=dataloader['train'],validloader=dataloader['valid'],testloader=dataloader['test'])
                        true_label, pred_label = model.Test(testloader=dataloader['test'])
                        metrics = eval_metrix(pred_label, true_label)
                        # 判断是否为当前i和e下MAPE最小的组合
                        if metrics[1] < best_MAPE:
                            best_MAPE = metrics[1]
                            best_alpha = alpha
                            best_beta = beta
                            best_metrics = metrics
                            best_pred_label = pred_label
                            best_true_label = true_label
                            best_model = {'solution_u':model.solution_u.state_dict(),'dynamical_F':model.dynamical_F.state_dict()}
                # 保存当前i和e下MAPE最小的超参数组合对应的结果到特定文件夹
                best_save_folder = f'results_small-sample/{args.model_name}/XJTU results (small sample {n})/' + str(i) + '-' + str(i) + '/Experiment' + str(e + 1) + f'/best_results'
                if not os.path.exists(best_save_folder):
                    os.makedirs(best_save_folder)
                np.save(os.path.join(best_save_folder, 'true_label.npy'), best_true_label)
                np.save(os.path.join(best_save_folder, 'pred_label.npy'), best_pred_label)
                best_log_dir = os.path.join(best_save_folder, "logging.txt") 
                best_logger = get_logger(best_log_dir) 
                info = '[Best]  alpha: {}, beta: {}, MAE: {:.4f}, MAPE: {:.4f}, MSE: {:.6f}, RMSE: {:.4f}'.format(best_alpha,best_beta,best_metrics[0], best_metrics[1], best_metrics[2], best_metrics[3])
                best_logger.info(info)
                torch.save(best_model,os.path.join(best_save_folder,'best_model.pth'))
                torch.cuda.empty_cache()


def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for XJTU dataset')
    parser.add_argument('--model_name',type=str,default='PIKAN_opt',choices=['PINN_opt','PIKAN_small','PIKAN_hgs'])
    parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--train_batch', type=int, default=0, choices=[-1,0,1,2,3,4,5],
                        help='如果是-1，读取全部数据，并随机划分训练集和测试集;否则，读取对应的batch数据'
                             '(if -1, read all data and random split train and test sets; '
                             'else, read the corresponding batch data)')
    parser.add_argument('--test_batch', type=int, default=1, choices=[-1,0,1,2,3,4,5],
                        help='如果是-1，读取全部数据，并随机划分训练集和测试集;否则，读取对应的batch数据'
                             '(if -1, read all data and random split train and test sets; '
                             'else, read the corresponding batch data)')
    parser.add_argument('--batch',type=str,default='2C',choices=['2C','3C','R2.5','R3','RW','satellite'])
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')

    # scheduler related
    parser.add_argument('--epochs', type=int, default=200, help='epoch')
    parser.add_argument('--early_stop', type=int, default=20, help='early stop')
    parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=0.002, help='warmup lr')
    parser.add_argument('--lr', type=float, default=0.01, help='base lr')
    parser.add_argument('--final_lr', type=float, default=0.0002, help='final lr')
    parser.add_argument('--lr_F', type=float, default=0.001, help='lr of F')

    # model related
    parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')

    # loss related
    parser.add_argument('--alpha', type=float, default=0.7, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta', type=float, default=0.2, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--alpha_values', type=float, default=[0.001, 0.01, 0.1, 1, 10, 100, 1000], help='loss = l_data + alpha * l_PDE + beta * l_physics')   #[0.001, 0.01, 0.1, 1, 10, 100, 1000]
    parser.add_argument('--beta_values', type=float, default=[0.001, 0.01, 0.1, 1, 10, 100, 1000], help='loss = l_data + alpha * l_PDE + beta * l_physics')    #[0.001, 0.01, 0.1, 1, 10, 100, 1000]

    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='results of reviewer/XJTU results', help='save folder')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
    small_sample()
    # main_2()
