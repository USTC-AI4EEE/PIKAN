import torch
import torch.nn as nn
import numpy as np
import os
from utils.util import AverageMeter,get_logger
from Model.Compare_Models import MLP,CNN,kan,KAN_small,Transformer,KAN_medium
from Model.PINN_opt import LR_Scheduler
from dataloader.dataloader import TJUdata
import argparse
from utils.util import set_seed


class Trainer():
    def __init__(self,model,train_loader,valid_loader,test_loader,args):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.args = args
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.save_dir = args.save_folder
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.epochs = args.epochs
        self.logger = get_logger(os.path.join(args.save_folder,args.log_dir))


        self.loss_meter = AverageMeter()
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=args.warmup_lr)
        self.scheduler = LR_Scheduler(optimizer=self.optimizer,
                                      warmup_epochs=args.warmup_epochs,
                                      warmup_lr=args.warmup_lr,
                                      num_epochs=args.epochs,
                                      base_lr=args.lr,
                                      final_lr=args.final_lr)

    def clear_logger(self):
        self.logger.removeHandler(self.logger.handlers[0])
        self.logger.handlers.clear()

    def train_one_epoch(self,epoch):
        self.model.train()
        self.loss_meter.reset()
        for (x1,_,y1,_) in self.train_loader:
            x1 = x1.to(self.device)
            y1 = y1.to(self.device)
            y_pred = self.model(x1)
            loss = self.loss_func(y_pred,y1)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.loss_meter.update(loss.item())
        info = '[Train] epoch:{}, data loss:{:.6f}'.format(epoch,self.loss_meter.avg)
        self.logger.info(info)
        return self.loss_meter.avg

    def valid(self,epoch):
        self.model.eval()
        self.loss_meter.reset()
        with torch.no_grad():
            for (x1,_,y1,_) in self.valid_loader:
                x1 = x1.to(self.device)
                y1 = y1.to(self.device)
                y_pred = self.model(x1)
                loss = self.loss_func(y_pred,y1)
                self.loss_meter.update(loss.item())
        info = '[Valid] epoch:{}, data loss:{:.6f}'.format(epoch,self.loss_meter.avg)
        self.logger.info(info)
        return self.loss_meter.avg

    def test(self):
        self.model.eval()
        self.loss_meter.reset()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for (x1,_,y1,_) in self.test_loader:
                x1 = x1.to(self.device)
                y_pred = self.model(x1)
                true_label.append(y1.cpu().detach().numpy())
                pred_label.append(y_pred.cpu().detach().numpy())
        true_label = np.concatenate(true_label,axis=0)
        pred_label = np.concatenate(pred_label,axis=0)
        if self.save_dir is not None:
            np.save(os.path.join(self.save_dir,'true_label.npy'),true_label)
            np.save(os.path.join(self.save_dir,'pred_label.npy'),pred_label)
        return true_label,pred_label

    def train(self):
        min_loss = 100
        early_stop = 0
        for epoch in range(1,self.epochs+1):
            early_stop += 1
            train_loss = self.train_one_epoch(epoch)
            current_lr = self.scheduler.step()
            valid_loss = self.valid(epoch)
            if valid_loss < min_loss and self.test_loader is not None:
                min_loss = valid_loss
                true_label,pred_label = self.test()
                early_stop = 0
            if early_stop > 10:
                break
        self.clear_logger()


def load_model(args):
    if args.model == 'MLP':
        model = MLP()
    elif args.model == 'CNN':
        model = CNN()
    elif args.model == 'KAN':
        model = kan()
    elif args.model == 'KAN_small':
        model = KAN_small()
    elif args.model == 'Transformer':
        model = Transformer()
    elif args.model == 'KAN_medium':
        model = KAN_medium()
    return model


def load_TJU_data_initial(args,small_sample=None):  # 这是PINN4SOH原文代码，但是这个代码读取文件时会因设备而异，故需要对此进行修改
    root = 'Data/TJU data'
    data = TJUdata(root=root, args=args)
    train_list = []
    test_list = []

    # 序号的个位数字是5或者9的是测试集，其他的是训练集
    # The numbers whose units digit is 5 or 9 are test set, and the others are training set
    mod = [(5,9),(4,8),(5,9)]
    if args.in_same_batch:
        batchs = os.listdir(root)
        batch = batchs[args.tju_batch]
        batch_root = os.path.join(root,batch)
        files = os.listdir(batch_root)
        for i,f in enumerate(files):
            # 判断i的个位数字是否为5或者9 (judge whether the units digit of i is 5 or 9)
            id = i + 1
            if id % 10 == mod[args.tju_batch][0] or id % 10 == mod[args.tju_batch][1]:
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
        train_loader = data.read_one_batch(args.tju_train_batch)
        test_loader = data.read_one_batch(args.tju_test_batch)
        dataloader = {'train': train_loader['train_2'],
                      'valid': train_loader['valid_2'],
                      'test': test_loader['test_3']}
    return dataloader


def load_TJU_data(args,small_sample=None):  # 这是修改后的代码，测试集选取的电池与原文结果文件里保持一致
    root = 'Data/TJU data'
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
        batch = batchs[args.tju_batch]
        batch_root = os.path.join(root,batch)
        files = os.listdir(batch_root)
        for f in files:
            if f[:-4] in test_id[args.tju_batch]:
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
        train_loader = data.read_one_batch(args.tju_train_batch)
        test_loader = data.read_one_batch(args.tju_test_batch)
        dataloader = {'train': train_loader['train_2'],
                      'valid': train_loader['valid_2'],
                      'test': test_loader['test_3']}
    return dataloader


def get_args():
    parser = argparse.ArgumentParser('The parameters of Comparision methods')
    parser.add_argument('--model',type=str,default='MLP',choices=['MLP','CNN','KAN'])
    parser.add_argument('--dataset',type=str,default='XJTU',choices=['XJTU','HUST','MIT','TJU'])
    parser.add_argument('--normalization_method',type=str, default='min-max', help='min-max,z-score')

    # XJTU data
    parser.add_argument('--xjtu_batch',type=str,default='2C',choices=['2C','3C','R2.5','R3','RW','satellite'])

    # TJU data
    parser.add_argument('--in_same_batch',type=bool,default=True)
    parser.add_argument('--tju_batch',type=int,default=0,choices=[0,1,2])
    parser.add_argument('--tju_train_batch',type=int,default=-1, choices=[-1,0,1,2])
    parser.add_argument('--tju_test_batch',type=int,default=-1, choices=[-1,0,1,2])

    # scheduler related
    parser.add_argument('--epochs', type=int, default=200, help='epoch')
    parser.add_argument('--early_stop', type=int, default=20, help='early stop')
    parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=2e-3, help='warmup lr')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--final_lr', type=float, default=2e-4, help='final lr')
    parser.add_argument('--lr_F', type=float, default=5e-4, help='lr of F')

    parser.add_argument('--save_folder',type=str,default='results/')
    parser.add_argument('--log_dir',type=str,default='logging.txt')
    parser.add_argument('--batch_size',type=int,default=512)

    args = parser.parse_args()
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    return args


def main_2():
    args = get_args()
    tju_batch = [0,1,2]
    setattr(args,'model','KAN_small') # select model: MLP or CNN or KAN
    setattr(args,'dataset','TJU') # select dataset: 'TJU'
    for i in [2]:
        setattr(args,'tju_batch',tju_batch[i])
        for e in range(10):
            set_seed(e) 
            setattr(args,'save_folder',os.path.join(f'results_soh-estimation/{args.model}/',f'{args.dataset}-{args.model} results/{i}-{i}/Experiment{e+1}'))
            if not os.path.exists(args.save_folder):
                os.makedirs(args.save_folder)
            model = load_model(args)
            tju_dataset = torch.load('Data/TJU_Data_Var2_batch_2.pt')
            trainer = Trainer(model,tju_dataset['train_loader'],tju_dataset['valid_loader'],tju_dataset['test_loader'],args)
            trainer.train()


def main():
    args = get_args()
    tju_batch = [0,1,2]
    setattr(args,'model','KAN_medium') # select model: MLP or CNN or KAN
    setattr(args,'dataset','TJU') # select dataset: 'TJU'
    for i in range(len(tju_batch)):
        setattr(args,'tju_batch',tju_batch[i])
        for e in range(10):
            set_seed(e) 
            setattr(args,'save_folder',os.path.join(f'results_soh-estimation/{args.model}/',f'{args.dataset}-{args.model} results/{i}-{i}/Experiment{e+1}'))
            if not os.path.exists(args.save_folder):
                os.makedirs(args.save_folder)
            model = load_model(args)
            data_loader = load_TJU_data(args)
            trainer = Trainer(model,data_loader['train'],data_loader['valid'],data_loader['test'],args)
            trainer.train()


def small_sample():
    args = get_args()
    batch = 2
    setattr(args,'model','KAN_medium') # select model: MLP or CNN or KAN
    setattr(args,'dataset','TJU') # select dataset: 'TJU'
    for num_battery in [1,2]:   # [1,2]
        for e in range(10):
            set_seed(e) 
            setattr(args,'in_same_batch',True)
            setattr(args,'tju_batch',batch)
            setattr(args,'save_folder',f'results_small-sample/{args.model}/TJU results (small sample {num_battery})/{batch}-{batch}/Experiment{e+1}')
            setattr(args, 'batch_size', 128)
            if not os.path.exists(args.save_folder):
                os.makedirs(args.save_folder)
            data_loader = load_TJU_data(args,small_sample=num_battery)
            model = load_model(args)
            trainer = Trainer(model,data_loader['train'],data_loader['valid'],data_loader['test'],args)
            trainer.train()


if __name__ == '__main__':
    main()
    small_sample()
    main_2()






