from dataloader.dataloader import TJUdata
from Model.SA_PIKAN import SA_PIKAN
from Model.PIKAN_AdpBal import PIKAN_AdpBal
from Model.PIKAN_PCGrad import PIKAN_PCGrad
from Model.PIKAN_Sum import PIKAN_Sum
from Model.PIKAN import PIKAN
from Model.PIMLP_PCGrad import PIMLP_PCGrad
from Model.PIMLP_Sum import PIMLP_Sum
from Model.PIMLP import PIMLP
from Model.PIMLP_v1 import PIMLP_v1
from Model.PIMultKAN import PIMultKAN
from Model.PIChebyKAN import PIChebyKAN
from Model.PIKAN_opt import PIKAN_opt
from Model.VerhulstPIKAN import VerhulstPIKAN
from Model.VerhulstPINN import VerhulstPINN
from Model.VerhulstPIKAN_v1 import VerhulstPIKAN_v1
import argparse
import os
from assistant import get_gpus_memory_info,set_seed
import numpy as np
import torch
from utils.util import eval_metrix,get_logger
def load_TJU_data_initial(args,small_sample=None):  # 这是PINN4SOH原文代码，但是这个代码读取文件时会因设备而异，故需要对此进行修改
    root = 'data/TJU data'
    data = TJUdata(root=root, args=args)
    train_list = []
    test_list = []

    # 序号的个位数字是5或者9的是测试集，其他的是训练集
    # The numbers whose units digit is 5 or 9 are test set, and the others are training set
    mod = [(5,9),(4,8),(5,9)]
    if args.in_same_batch:
        batchs = os.listdir(root)
        batch = batchs[args.batch]
        batch_root = os.path.join(root,batch)
        files = os.listdir(batch_root)
        for i,f in enumerate(files):
            # 判断i的个位数字是否为5或者9 (judge whether the units digit of i is 5 or 9)
            id = i + 1
            if id % 10 == mod[args.batch][0] or id % 10 == mod[args.batch][1]:
                test_list.append(os.path.join(batch_root,f))
                print(f)
            else:
                train_list.append(os.path.join(batch_root,f))
        if small_sample is not None:
            train_list = train_list[:small_sample]
        train_loader = data.read_all(specific_path_list=train_list)
        test_loader = data.read_all(specific_path_list=test_list)
        dataloader = {'train': train_loader['train_2'],
                      'valid': train_loader['valid_2'],
                      'test': test_loader['test_3']}
    else: # 如果训练集和测试集不在同一个batch中，则一个batch用来训练，另一个batch用来测试
        # (If the training set and test set are not in the same batch,
        # one batch is used for training and the other batch is used for testing)
        batchs = os.listdir(root)
        train_loader = data.read_one_batch(args.train_batch)
        test_loader = data.read_one_batch(args.test_batch)
        dataloader = {'train': train_loader['train_2'],
                      'valid': train_loader['valid_2'],
                      'test': test_loader['test_3']}
    return dataloader

def load_TJU_data(args,small_sample=None):  # 这是修改后的代码，测试集选取的电池与原文结果文件里保持一致
    root = 'data/TJU data'
    data = TJUdata(root=root, args=args)
    train_list = []
    test_list = []
    test_id = [['CY25-025_1-#5','CY25-05_1-#10','CY25-05_1-#16','CY25-05_1-#2','CY25-05_1-#8','CY25-1_1-#3','CY25-1_1-#9',
                'CY45-05_1-#1','CY45-05_1-#15','CY45-05_1-#19','CY45-05_1-#24','CY45-05_1-#28','CY45-05_1-#8'],# batch1
                ['CY25-05_1-#12','CY25-05_1-#16','CY25-05_1-#21','CY25-05_1-#4','CY35-05_1-#1','CY45-05_1-#1',
                 'CY45-05_1-#15','CY45-05_1-#19','CY45-05_1-#24','CY45-05_1-#28','CY45-05_1-#8'],# batch2
                 ['CY25-05_2-#2','CY25-05_4-#3']]# batch3
    if args.in_same_batch:
        batchs = os.listdir(root)
        batch = batchs[args.batch]
        batch_root = os.path.join(root,batch)
        files = os.listdir(batch_root)
        for f in files:
            if f[:-4] in test_id[args.batch]:
                test_list.append(os.path.join(batch_root,f))
                print(f)
            else:
                train_list.append(os.path.join(batch_root,f))
        if small_sample is not None:
            train_list = train_list[:small_sample]
        train_loader = data.read_all(specific_path_list=train_list)
        test_loader = data.read_all(specific_path_list=test_list)
        dataloader = {'train': train_loader['train_2'],
                      'valid': train_loader['valid_2'],
                      'test': test_loader['test_3']}
    else: # 如果训练集和测试集不在同一个batch中，则一个batch用来训练，另一个batch用来测试
        # (If the training set and test set are not in the same batch,
        # one batch is used for training and the other batch is used for testing)
        batchs = os.listdir(root)
        train_loader = data.read_one_batch(args.train_batch)
        test_loader = data.read_one_batch(args.test_batch)
        dataloader = {'train': train_loader['train_2'],
                      'valid': train_loader['valid_2'],
                      'test': test_loader['test_3']}
    return dataloader

def load_model(args):
    if args.model_name == 'SA_PIKAN':
        model = SA_PIKAN(args)
    elif args.model_name == 'PIKAN_AdpBal':
        model = PIKAN_AdpBal(args)
    elif args.model_name == 'PIKAN_PCGrad':
        model = PIKAN_PCGrad(args)
    elif args.model_name == 'PIKAN_Sum':
        model = PIKAN_Sum(args)
    elif args.model_name == 'PIKAN':
        model = PIKAN(args)
    elif args.model_name == 'PIMLP_PCGrad':
        model = PIMLP_PCGrad(args)
    elif args.model_name == 'PIMLP_Sum':
        model = PIMLP_Sum(args)
    elif args.model_name == 'PIMLP':
        model = PIMLP(args)
    elif args.model_name == 'PIMLP_v1':
        model = PIMLP_v1(args)
    elif args.model_name == 'PIMultKAN':
        model = PIMultKAN(args)
    elif args.model_name == 'PIChebyKAN':
        model = PIChebyKAN(args)
    elif args.model_name == 'PIKAN_opt':
        model = PIKAN_opt(args)
    elif args.model_name == 'VerhulstPIKAN':
        model = VerhulstPIKAN(args)
    elif args.model_name == 'VerhulstPINN':
        model = VerhulstPINN(args)
    elif args.model_name == 'VerhulstPIKAN_v1':
        model = VerhulstPIKAN_v1(args)
    return model

def main():
    args = get_args()
    setattr(args,'model_name','VerhulstPIKAN_v1')
    batchs = [2]       
    for batch in batchs:
        setattr(args, 'in_same_batch', True)
        setattr(args, 'batch', batch)
        for e in range(10):
            set_seed(e)         
            if args.in_same_batch:
                save_folder = f'results_soh-estimation/{args.model_name}/TJU results/' + str(batch) + '-' + str(batch) + '/Experiment' + str(e + 1)
            else:
                save_folder = f'results_soh-estimation/{args.model_name}/TJU results/' + str(args.train_batch) + '-' + str(args.test_batch) + '/Experiment' + str(e + 1)
            setattr(args, "save_folder", save_folder)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            log_dir = 'logging.txt'
            setattr(args, "save_folder", save_folder)
            setattr(args, "log_dir", log_dir)
            dataloader = load_TJU_data(args)   
            model = load_model(args)
            model.Train(trainloader=dataloader['train'],validloader=dataloader['valid'],testloader=dataloader['test'])
            torch.cuda.empty_cache()

def small_sample():
    args = get_args()
    setattr(args,'model_name','VerhulstPIKAN_v1')
    batchs = [0,1,2]
    for n in [1,2]:
        for i in [2]:
            batch = batchs[i]
            setattr(args, 'batch', batch)
            setattr(args,'batch_size',128)
            for e in range(10):
                set_seed(e)         
                save_folder = f'results_small-sample/{args.model_name}/TJU results (small sample {n})/' + str(batch) + '-' + str(batch) + '/Experiment' + str(e + 1) 
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                log_dir = 'logging.txt'
                setattr(args, "save_folder", save_folder)
                setattr(args, "log_dir", log_dir)
                dataloader = load_TJU_data(args,small_sample=n)   
                model = load_model(args)
                model.Train(trainloader=dataloader['train'],validloader=dataloader['valid'],testloader=dataloader['test'])
                torch.cuda.empty_cache()

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for TJU dataset')
    parser.add_argument('--model_name',type=str,default='PIKAN',choices=['SA_PIKAN','PIKAN_AdpBal','PIKAN_PCGrad','PIMLP_PCGrad','PIKAN_Sum','PIMLP_Sum','PIKAN','PIMLP'])
    parser.add_argument('--data', type=str, default='TJU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--in_same_batch', type=bool, default=True, help='训练集和测试集是否在同一个batch中(whether train and test sets are in the same batch)')
    parser.add_argument('--train_batch', type=int, default=-1, choices=[-1,0,1,2],
                        help='如果是-1，读取全部数据，并随机划分训练集和测试集;否则，读取对应的batch数据'
                             '(if -1, read all data and random split train and test sets; '
                             'else, read the corresponding batch data)')
    parser.add_argument('--test_batch', type=int, default=-1, choices=[-1,0,1,2],
                        help='如果是-1，读取全部数据，并随机划分训练集和测试集;否则，读取对应的batch数据'
                             '(if -1, read all data and random split train and test sets; '
                             'else, read the corresponding batch data)')
    parser.add_argument('--batch',type=int,default=0,choices=[0,1,2])
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
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
    parser.add_argument('--lr_cos',type=int,default=0.001,help='lr of cos_sim')
    parser.add_argument('--lr_w',type=int,default=0.001,help='lr of loss weights')
    parser.add_argument('--mode',type=str,default='AdpBal',help=['Baseline','Sum','AdpBal'])
    parser.add_argument('--alpha', type=float, default=10, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta', type=float, default=0.1, help='loss = l_data + alpha * l_PDE + beta * l_physics')

    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='results of reviewer/XJTU results', help='save folder')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # pass
    main()
    # small_sample()
