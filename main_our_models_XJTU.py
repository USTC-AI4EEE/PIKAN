from dataloader.dataloader import XJTUdata
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
import argparse
import os
from assistant import get_gpus_memory_info,set_seed
import numpy as np
import torch
from utils.util import eval_metrix,get_logger
def load_data(args,small_sample=None):
    root = 'data/XJTU data'
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
    return model

def main():
    args = get_args()
    setattr(args,'model_name','VerhulstPINN')
    batchs = ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']
    for i in range(6):
        batch = batchs[i]
        setattr(args, 'batch', batch)
        for e in range(10):
            set_seed(e)         
            save_folder = f'results_soh-estimation/{args.model_name}/XJTU results/' + str(i) + '-' + str(i) + '/Experiment' + str(e + 1)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            log_dir = 'logging.txt'
            setattr(args, "save_folder", save_folder)
            setattr(args, "log_dir", log_dir)
            dataloader = load_data(args)   
            model = load_model(args)
            model.Train(trainloader=dataloader['train'],validloader=dataloader['valid'],testloader=dataloader['test'])
            torch.cuda.empty_cache()

def small_sample():
    args = get_args()
    setattr(args,'model_name','VerhulstPINN')
    batchs = ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']
    for n in [1,2,3,4]:
        for i in range(6):
            batch = batchs[i]
            setattr(args, 'batch', batch)
            setattr(args,'batch_size',128)
            setattr(args,'alpha',0.5)
            setattr(args,'beta',10)
            for e in range(10):
                set_seed(e)         
                save_folder = f'results_small-sample/{args.model_name}/XJTU results (small sample {n})/' + str(i) + '-' + str(i) + '/Experiment' + str(e + 1)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                log_dir = 'logging.txt'
                setattr(args, "save_folder", save_folder)
                setattr(args, "log_dir", log_dir)
                dataloader = load_data(args,small_sample=n)   
                model = load_model(args)
                model.Train(trainloader=dataloader['train'],validloader=dataloader['valid'],testloader=dataloader['test'])
                torch.cuda.empty_cache()

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for XJTU dataset')
    parser.add_argument('--model_name',type=str,default='PIKAN',choices=['SA_PIKAN','PIKAN_AdpBal','PIKAN_PCGrad','PIMLP_PCGrad','PIKAN_Sum','PIMLP_Sum','PIKAN','PIMLP'])
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
    parser.add_argument('--lr_cos',type=int,default=0.001,help='lr of cos_sim')
    parser.add_argument('--lr_w',type=int,default=0.001,help='lr of loss weights')
    parser.add_argument('--mode',type=str,default='AdpBal',help=['Baseline','Sum','AdpBal'])
    parser.add_argument('--alpha', type=float, default=0.7, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta', type=float, default=0.2, help='loss = l_data + alpha * l_PDE + beta * l_physics')

    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='results of reviewer/XJTU results', help='save folder')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
    small_sample()
