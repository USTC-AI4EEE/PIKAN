'''
XJTU数据集的结果分析

English:
    This file is used to analyze the results of the XJTU dataset.
'''
import pandas as pd
import numpy as np
import os
from sklearn import metrics


def eval_metrix(true_label,pred_label):
    MAE = metrics.mean_absolute_error(true_label,pred_label)
    MAPE = metrics.mean_absolute_percentage_error(true_label,pred_label)
    MSE = metrics.mean_squared_error(true_label,pred_label)
    RMSE = np.sqrt(metrics.mean_squared_error(true_label,pred_label))
    return [MAE,MAPE,MSE,RMSE]

class Results:
    def __init__(self,root='../results/Ours/XJTU results/'):
        self.root = root
        self.experiments = os.listdir(root)
        self.subfolder = None
        self.log_dir = None
        self.pred_label = None
        self.true_label = None
        self._update_experiments(1)

    def _update_experiments(self,train_batch=0,test_batch=1,experiment=1,subfolder='best_results'):
        self.subfolder = f'{train_batch}-{test_batch}/Experiment{experiment}/' + subfolder
        self.log_dir = os.path.join(self.root, self.subfolder,'logging.txt')
        self.pred_label = os.path.join(self.root, self.subfolder,'pred_label.npy')
        self.true_label = os.path.join(self.root, self.subfolder,'true_label.npy')

    def parser_log(self):
        '''
        解析train过程中产生的log文件，获取里面的数据
        English:
            Parse the log file generated during the training process to obtain the data
        :return: dict
        '''
        data_dict = {}

        with open(self.log_dir, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if 'CRITICAL' in line:
                params = line.split('\t')[-1].split('\n')[0]
                k, v = params.split(':')
                data_dict[k] = v

        train_total_loss = []
        valid_data_loss = []

        test_mse = []
        test_epoch = []

        for i in range(len(lines)):
            line = lines[i]
            if '[train] epoch:1 iter:1 data' in line:
                train_total_loss.append(float(line.split('total loss:')[1].split('\n')[0]))
            elif '[Train]' in line:
                train_total_loss.append(float(line.split('total loss:')[1].split('\n')[0]))
            elif '[Valid]' in line:
                valid_data_loss.append(float(line.split('MSE:')[1].split('\n')[0]))
            elif '[Test]' in line:
                test_mse.append(float(line.split('MSE:')[1].split(',')[0]))
                test_epoch.append(int(lines[i - 1].split('epoch:')[1].split(',')[0]))

        data_dict['train_total_loss'] = train_total_loss
        data_dict['valid_data_loss'] = valid_data_loss
        data_dict['test_mse'] = test_mse
        data_dict['test_epoch'] = test_epoch

        line1 = lines[1]
        if '.csv' in line1:
            line = line1[1:-2]
            line_list = line.replace('Data/XJTU data/', '').replace('.csv','').replace('\'','').split(', ')
            data_dict['IDs_1'] = line_list

        line2 = lines[3]
        if '.csv' in line2:
            line = line2[1:-2]
            line_list = line.replace('Data/XJTU data/', '').replace('.csv', '').replace('\'', '').split(', ')
            data_dict['IDs_2'] = line_list

        return data_dict

    def parser_label(self):
        '''
        解析预测结果
        English:
            Parse the prediction results
        :return:
        '''
        pred_label = np.load(self.pred_label).reshape(-1)
        true_label = np.load(self.true_label).reshape(-1)

        # 用来保存每个电池的预测结果
        # Save the prediction results of each battery
        pred_label_list = []
        true_label_list = []
        MAE_list = []
        MAPE_list = []
        MSE_list = []
        RMSE_list = []

        diff = np.diff(true_label)
        split_point = np.where(diff > 0.05)[0]
        local_minima = np.concatenate((split_point, [len(true_label)]))

        start = 0
        end = 0
        for i in range(len(local_minima)):
            end = local_minima[i]
            pred_i = pred_label[start:end]
            true_i = true_label[start:end]
            [MAE_i, MAPE_i, MSE_i, RMSE_i] = eval_metrix(pred_i, true_i)
            # print('battery {} MAE:{:.4f}, MAPE:{:.4f}, MSE:{:.6f}, RMSE:{:.4f}:{:.4f}'.format(i + 1, MAE_i, MAPE_i,
            #                                                                                       MSE_i, RMSE_i))
            start = end + 1

            pred_label_list.append(pred_i)
            true_label_list.append(true_i)
            MAE_list.append(MAE_i)
            MAPE_list.append(MAPE_i)
            MSE_list.append(MSE_i)
            RMSE_list.append(RMSE_i)


        results_dict = {}
        results_dict['pred_label'] = pred_label_list
        results_dict['true_label'] = true_label_list
        results_dict['MAE'] = MAE_list
        results_dict['MAPE'] = MAPE_list
        # results_dict['MSE'] = MSE_list
        results_dict['RMSE'] = RMSE_list

        return results_dict

    def get_test_results(self,train=0,test=1,e=1,s='best_results'):
        '''
        解析训练和测试数据中的电池id
        English:
            Parse the battery id in the training and test sets
        :param e: experiment id
        :return:
        '''
        self._update_experiments(train_batch=train,test_batch=test,experiment=e,subfolder=s)
        # log_dict = self.parser_log()
        results_dict = self.parser_label()
        # results_dict['channel'] = log_dict['IDs_2']
        return results_dict

    def get_battery_mean(self,train_batch,test_batch,total_experiment=10,subfolder='best_results'):
        df_mean_values = []
        for e in range(1, 1+total_experiment):
            res = results.get_test_results(train=train_batch, test=test_batch, e=e,s=subfolder)
            df = pd.DataFrame(res)
            df = df[['MAE', 'MAPE', 'RMSE']]
            df_i_mean = df.mean(axis=0)
            df_mean_values.append(df_i_mean.values)
        df_mean_values = np.array(df_mean_values)
        df_mean = pd.DataFrame(df_mean_values, columns=['MAE', 'MAPE', 'RMSE'])
        df_mean.insert(0, column='experiment', value=np.arange(1, 1+total_experiment))
        # 最后添加所有样本的均值 (add the mean of all samples)
        mean = df_mean.mean(axis=0)

        # df_mean = df_mean.append(mean, ignore_index=True)
        print('-'*50)
        print(f'batch {train_batch+1}')
        print(f'mean:  MAPE:{mean[2]:.4f}, RMSE:{mean[3]:.4f}')

        return df_mean

    def get_group_mean(self,train_batch,test_batch,total_experiment=10,total_group=49):
        s_df_mean_values = []
        for group in range(1, 1+total_group):
            s = 'group' + str(group)
            df_mean_values = []
            for e in range(1, 1+total_experiment):
                res = results.get_test_results(train=train_batch, test=test_batch, e=e, s=s)
                df = pd.DataFrame(res)
                df = df[['MAE', 'MAPE', 'RMSE']]
                df_i_mean = df.mean(axis=0)
                df_mean_values.append(df_i_mean.values)
            df_mean_values = np.array(df_mean_values)
            df_mean = pd.DataFrame(df_mean_values, columns=['MAE', 'MAPE', 'RMSE'])
            df_mean.insert(0, column='experiment', value=np.arange(1, 1+total_experiment))
            # 只计算性能指标列的平均值，不包括experiment列
            s_df_mean = df_mean[['MAE', 'MAPE', 'RMSE']].mean(axis=0)
            s_df_mean_values.append(s_df_mean.values)
        s_df_mean_values = np.array(s_df_mean_values)
        s_df_mean = pd.DataFrame(s_df_mean_values, columns=['MAE', 'MAPE', 'RMSE'])
        s_df_mean.insert(0, column='group', value=np.arange(1, 1+total_group))
        return s_df_mean

    def get_experiments_mean(self,train_batch=0,test_batch=0,total_experiment=10):
        '''
        分别获取每个测试电池在所有实验中的平均值
        English:
            Get the average value of each test battery in all experiments
        :return: dataframe，每一行是一个电池在10次实验中的平均值 (each row is the average value of a battery in 10 experiments)
        '''
        df_value_list = []
        for i in range(1,1+total_experiment):
            res = self.get_test_results(train_batch,test_batch,i)
            df = pd.DataFrame(res)
            df = df[['channel','MAE', 'MAPE', 'RMSE']]
            df = df.sort_values(by='channel')
            df.reset_index(drop=True, inplace=True)
            df_value_list.append(df[['MAE','MAPE','RMSE']].values)
        channel = df['channel']
        columns = ['MAE', 'MAPE', 'RMSE']

        np_array = np.array(df_value_list)
        np_mean = np.mean(np_array,axis=0)
        df_mean = pd.DataFrame(np_mean,columns=columns)
        df_mean.insert(0,column='channel',value=channel)
        df_mean['channel'] = df_mean['channel'].apply(lambda x: x[-9:])
        #print(df_mean)
        return df_mean
    

if __name__ == '__main__':

    model_name = 'PIKAN_opt' # PIKAN_opt or PIKAN_medium or PINN_opt or PIKAN_small

    # XJTU soh-estimation
    root = f'results_soh-estimation/{model_name}/XJTU results/'
    save_folder = f'results_soh-estimation/processed_results/{model_name}/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    writer = pd.ExcelWriter(f'results_soh-estimation/processed_results/{model_name}/{model_name}_XJTU_results_group_mean.xlsx')
    results = Results(root)
    for batch in range(6):
        df_group_mean = results.get_group_mean(train_batch=batch,test_batch=batch,total_experiment=10,total_group=49)
        df_group_mean.to_excel(writer,sheet_name='group_mean_{}'.format(batch),index=False)
    writer._save()

    # XJTU small-sample
    for n in [1,2,3,4]:
        root = f'results_small-sample/{model_name}/XJTU results (small sample {n})/'
        save_folder = f'results_small-sample/processed_results/{model_name}/'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        writer = pd.ExcelWriter(f'results_small-sample/processed_results/{model_name}/{model_name}_XJTU_results_small_sample_{n}_group_mean.xlsx')
        results = Results(root)
        for batch in [0]:
            df_group_mean = results.get_group_mean(train_batch=batch,test_batch=batch,total_experiment=10,total_group=49)
            df_group_mean.to_excel(writer,sheet_name='group_mean_{}'.format(batch),index=False)
        writer._save()