from Model.PIKAN_opt import PIKAN_opt as PIKAN
import torch
import torch.nn as nn
from dataloader.dataloader import XJTUdata,TJUdata,MITdata,HUSTdata
from Model.Model import LR_Scheduler
import argparse
import os
import numpy as np
from utils.util import AverageMeter,eval_metrix,write_to_txt
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class AdaPIKAN(PIKAN):
    def __init__(self,args):
        super(AdaPIKAN, self).__init__(args)

        self.load_model(model_path=args.pretrain_model)
        self.ada_optimizer = torch.optim.Adam(self.solution_u.parameters(),lr=args.adaptation_lr)

    def adaptation_one_epoch(self,epoch,dataloader):
        self.solution_u.train()
        loss1_meter = AverageMeter()
        loss2_meter = AverageMeter()
        loss3_meter = AverageMeter()
        for iter,(x1,x2,y1,y2) in enumerate(dataloader):
            x1,x2,y1,y2 = x1.to(device),x2.to(device),y1.to(device),y2.to(device)
            u1,f1 = self.forward(x1)
            u2,f2 = self.forward(x2)

            # data loss
            loss1 = 0.5*self.loss_func(u1,y1) + 0.5*self.loss_func(u2,y2)

            # PDE loss
            f_target = torch.zeros_like(f1)
            loss2 = 0.5*self.loss_func(f1,f_target) + 0.5*self.loss_func(f2,f_target)

            # physics loss  u2-u1<0, considering capacity regeneration effect
            loss3 = self.relu(torch.mul(u2-u1,y1-y2)).sum()

            # total loss
            loss = loss1 + self.alpha*loss2 + self.beta*loss3

            self.ada_optimizer.zero_grad()
            loss.backward()
            self.ada_optimizer.step()

            loss1_meter.update(loss1.item())
            loss2_meter.update(loss2.item())
            loss3_meter.update(loss3.item())
            debug_info = "[train] epoch:{} iter:{} data loss:{:.6f}, " \
                         "PDE loss:{:.6f}, physics loss:{:.6f}, " \
                         "total loss:{:.6f}".format(epoch,iter+1,loss1,loss2,loss3,loss.item())
            if epoch < 3:
                self.logger.debug(debug_info)

            if (iter+1) % 50 == 0:
                print("[epoch:{} iter:{}] data loss:{:.6f}, PDE loss:{:.6f}, physics loss:{:.6f}".format(epoch,iter+1,loss1,loss2,loss3))
        return loss1_meter.avg,loss2_meter.avg,loss3_meter.avg

    def Adaptation(self,trainloader,validloader=None,testloader=None):
        for param in self.dynamical_F.parameters(): # freeze the dynamical_F
            param.requires_grad = False

        min_valid_mse = 10
        valid_mse = 10
        early_stop = 0
        mae = 10
        for e in range(1, self.args.adaptation_epochs + 1):
            early_stop += 1
            loss1, loss2, loss3 = self.adaptation_one_epoch(e, trainloader)
            info = '[Train] epoch:{}, data loss:{:.6f}, ' \
                   'PDE loss:{:.6f}, ' \
                   'physics loss:{:.6f}, ' \
                   'total loss:{:.6f}'.format(e, loss1, loss2, loss3,
                                              loss1 + self.alpha * loss2 + self.beta * loss3)
            self.logger.info(info)
            if e % 1 == 0 and validloader is not None:
                valid_mse = self.Valid(validloader)
                info = '[Valid] epoch:{}, MSE: {}'.format(e, valid_mse)
                self.logger.info(info)
            if valid_mse < min_valid_mse and testloader is not None:
                min_valid_mse = valid_mse
                true_label, pred_label = self.Test(testloader)
                [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_label, true_label)
                info = '[Test] MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}'.format(MSE, MAE, MAPE, RMSE)
                self.logger.info(info)
                early_stop = 0

                ############################### save ############################################
                self.best_model = {'solution_u': self.solution_u.state_dict(),
                                   'dynamical_F': self.dynamical_F.state_dict()}
                if self.args.save_folder is not None:
                    np.save(os.path.join(self.args.save_folder, 'true_label.npy'), true_label)
                    np.save(os.path.join(self.args.save_folder, 'pred_label.npy'), pred_label)
                ##################################################################################
            if self.args.early_stop is not None and early_stop > self.args.early_stop:
                info = 'early stop at epoch {}'.format(e)
                self.logger.info(info)
                break
        self.clear_logger()
        if self.args.save_folder is not None:
            torch.save(self.best_model, os.path.join(self.args.save_folder, 'finetune model.pth'))

def load_XJTU_data(args,small_sample=None):
    root = 'data/XJTU data'
    batch_names= ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']
    batch_num = args.target_batch if args.target_data == 'XJTU' else args.source_batch
    batch = batch_names[batch_num]
    data = XJTUdata(root=root, args=args)
    train_list = []
    test_list = []
    files = os.listdir(root)
    for file in files:
        if batch in file:
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

def load_TJU_data(args,small_sample=None):
    root = 'data/TJU data'
    data = TJUdata(root=root, args=args)
    train_list = []
    test_list = []
    test_id = [['CY25-025_1-#5','CY25-05_1-#10','CY25-05_1-#16','CY25-05_1-#2','CY25-05_1-#8','CY25-1_1-#3','CY25-1_1-#9',
                'CY45-05_1-#1','CY45-05_1-#15','CY45-05_1-#19','CY45-05_1-#24','CY45-05_1-#28','CY45-05_1-#8'],# batch1
                ['CY25-05_1-#12','CY25-05_1-#16','CY25-05_1-#21','CY25-05_1-#4','CY35-05_1-#1','CY45-05_1-#1',
                 'CY45-05_1-#15','CY45-05_1-#19','CY45-05_1-#24','CY45-05_1-#28','CY45-05_1-#8'],# batch2
                 ['CY25-05_2-#2','CY25-05_4-#3']]# batch3

    batch_num = args.target_batch if args.target_data == 'TJU' else args.source_batch
    batchs = os.listdir(root)
    batch = batchs[batch_num]
    batch_root = os.path.join(root,batch)
    files = os.listdir(batch_root)
    for f in files:
        if f[:-4] in test_id[batch_num]:
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
    return dataloader

def load_MIT_data(args,small_sample=None):
    root = 'data/MIT data'
    train_list = []
    test_list = []
    for batch in ['2017-05-12','2017-06-30','2018-04-12']:
        batch_root = os.path.join(root,batch)
        files = os.listdir(batch_root)
        for f in files:
            id = int(f.split('-')[-1].split('.')[0])
            if id % 5 == 0:
                test_list.append(os.path.join(batch_root,f))
            else:
                train_list.append(os.path.join(batch_root,f))
    if small_sample is not None:
        train_list = train_list[:small_sample]
    data = MITdata(root=root,args=args)
    trainloader = data.read_all(specific_path_list=train_list)
    testloader = data.read_all(specific_path_list=test_list)
    dataloader = {'train':trainloader['train_2'],'valid':trainloader['valid_2'],'test':testloader['test_3']}

    return dataloader

def load_HUST_data(args,small_sample=None):
    test_id = ['1-4','1-8','2-4','2-8',
               '3-4','3-8','4-4','4-8',
               '5-4','5-7','6-4','6-8',
               '7-4','7-8','8-4','8-8',
               '9-4','9-8','10-4','10-8']
    data = HUSTdata(root='data/HUST data',args=args)
    train_list = []
    test_list = []
    files = os.listdir('data/HUST data')
    for f in files:
        if f[:-4] in test_id:
            test_list.append(f'data/HUST data/{f}')
        else:
            train_list.append(f'data/HUST data/{f}')
    if small_sample is not None:
        train_list = train_list[:small_sample]

    trainloader = data.read_all(specific_path_list=train_list)
    testloader = data.read_all(specific_path_list=test_list)
    dataloader = {'train':trainloader['train_2'],'valid':trainloader['valid_2'],'test':testloader['test_3']}

    return dataloader

def one_adaptation_task(args,source,target,source_batch=-1,target_batch=-1):
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    if source in ['XJTU','TJU']:
        model_dir = f'pretrained_models/{args.model_name}/model_{source}_{source_batch}.pth'
    else:
        model_dir = f'pretrained_models/{args.model_name}/model_{source}.pth'
    setattr(args,'pretrain_model',model_dir)
    setattr(args,'target_data',target)
    setattr(args,'target_batch',target_batch)

    # load data
    target_loader = eval(f'load_{target}_data')(args,small_sample=1)

    # load model
    model = AdaPIKAN(args)

    # Firstly, test source model in target domain
    true_label,pred_label = model.Test(target_loader['test'])
    [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_label, true_label)

    print('Before adaptation (source only):')
    print('MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}'.format(MSE, MAE, MAPE, RMSE))
    if args.log_dir is not None and args.save_folder is not None:
        save_name = os.path.join(args.save_folder,args.log_dir)
        info = 'Source only: {} -> {} | MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}'.format(source,target,MSE, MAE, MAPE, RMSE)
        write_to_txt(save_name,info)

    # adaptation
    model.Adaptation(trainloader=target_loader['train'],validloader=target_loader['valid'],testloader=target_loader['test'])

def FineTune_TJU2XJTU():
    args = get_args()
    lrs = [0.0004,0.01,0.0005,0.002,0.003,0.0006]
    source_batchs = [2,2,2,1,0,1]
    target_batchs = [0,1,2,3,4,5]
    for lr,sb,tb in zip(lrs,source_batchs,target_batchs):
        for experiment in range(10):
            setattr(args,'adaptation_lr',lr)
            setattr(args,'log_dir','logging.txt')
            setattr(args,'save_folder',f'results_fine-tuning/{args.model_name}/fine_tuning/TJU-XJTU/batch{tb}/Experiment{experiment + 1}')
            one_adaptation_task(args,source='TJU',target='XJTU',source_batch=sb,target_batch=tb)

def FineTune_XJTU2TJU():
    args = get_args()
    lrs = [0.003,0.002,0.002]
    source_batchs = [3,3,2]
    target_batchs = [0,1,2]
    for lr,sb,tb in zip(lrs,source_batchs,target_batchs):
        for experiment in range(10):
            setattr(args,'adaptation_lr',lr)
            setattr(args,'log_dir','logging.txt')
            setattr(args,'save_folder',f'results_fine-tuning/{args.model_name}/fine_tuning/XJTU-TJU/batch{tb}/Experiment{experiment + 1}')
            one_adaptation_task(args,source='XJTU',target='TJU',source_batch=sb,target_batch=tb)

def FineTune_HUST2MIT():
    args = get_args()
    for experiment in range(10):
        setattr(args,'adaptation_lr',0.005)
        setattr(args,'log_dir','logging.txt')
        setattr(args,'save_folder',f'results_fine-tuning/{args.model_name}/fine_tuning/HUST-MIT/Experiment{experiment + 1}')
        one_adaptation_task(args,source='HUST',target='MIT')

def FineTune_MIT2HUST():
    args = get_args()
    for experiment in range(10):
        setattr(args,'adaptation_lr',0.0002)
        setattr(args,'log_dir','logging.txt')
        setattr(args,'save_folder',f'results_fine-tuning/{args.model_name}/fine_tuning/MIT-HUST/Experiment{experiment + 1}')
        one_adaptation_task(args,source='MIT',target='HUST')

def FineTune():
    args = get_args()
    datasets = ['XJTU', 'TJU', 'HUST', 'MIT']
    batchs = [0, 2, -1, -1]
    for i, source in enumerate(datasets):
        for j, target in enumerate(datasets):
            if source in ['XJTU', 'TJU'] and target in ['XJTU', 'TJU']:
                continue
            if source in ['HUST', 'MIT'] and target in ['HUST', 'MIT']:
                continue
            sb = batchs[i]
            tb = batchs[j]

            for e in range(10):
                setattr(args, 'adaptation_lr', 0.001)
                setattr(args, 'log_dir', f'logging.txt')
                setattr(args, 'save_folder',
                        f'results_fine-tuning/{args.model_name}/fine_tuning/{source}-{target}/Experiment{e + 1}')
                one_adaptation_task(args, source=source, target=target, source_batch=sb, target_batch=tb)

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for fine-tuning')
    parser.add_argument('--model_name',type=str,default='PIKAN',choices=['PIKAN'])
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')

    # scheduler related
    parser.add_argument('--epochs', type=int, default=200, help='epoch')
    parser.add_argument('--early_stop', type=int, default=10, help='early stop')
    parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=0.002, help='warmup lr')
    parser.add_argument('--lr', type=float, default=0.01, help='base lr')
    parser.add_argument('--final_lr', type=float, default=0.0002, help='final lr')
    parser.add_argument('--lr_F', type=float, default=0.01, help='lr of F')

    # model related
    parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')

    # loss related
    parser.add_argument('--alpha', type=float, default=0.7, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta', type=float, default=0.2, help='loss = l_data + alpha * l_PDE + beta * l_physics')

    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='AdaPIKAN_test', help='save folder')

    # The AdaPIKAN class inherits the PIKAN_opt class, and the above parameters are all parameters of PIKAN_opt.
    # The following are the parameters of AdaPIKAN.
    # adaption related
    parser.add_argument('--pretrain_model', type=str, default=None, help='The saving path of the model trained in the source domain')
    parser.add_argument('--adaptation_lr', type=float, default=4e-4, help='adaption lr')
    parser.add_argument('--adaptation_epochs', type=int, default=200, help='adaption epochs')

    parser.add_argument('--target_data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--target_batch', type=int, default=-1, choices=[-1,0,1,2,3,4,5],
                        help='XJTU dataset is divided into 6 batches, and TJU dataset is divided into 3 batches. '
                             'If target_data is XJTU, the value range of target_batch is [-1,0,1,2,3,4,5];'
                             'If target_data is TJU, the value range of target_batch is [-1,0,1,2];'
                             'If it is other datasets, ignore target_batch')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    # FineTune_MIT2HUST()
    FineTune_HUST2MIT()
    # FineTune_TJU2XJTU()
    # FineTune_XJTU2TJU()
    FineTune()
    # pass
