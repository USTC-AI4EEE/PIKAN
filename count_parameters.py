from Model.PINN_opt import PINN_opt
from Model.PIKAN_opt import PIKAN_opt
from Model.PIKAN_hgs import PIKAN_hgs
from Model.PIKAN_small import PIKAN_small
from Model.Compare_Models import MLP,CNN,kan,KAN_small,Transformer,KAN_medium
import argparse


def count_parameters(model):    # 计算模型中需要进行梯度更新的参数的总数
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The model has {} trainable parameters'.format(count))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--batch', type=int, default=10, help='1,2,3')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='z-score', help='min-max,z-score')

    # scheduler 相关
    parser.add_argument('--epochs', type=int, default=1, help='epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=5e-4, help='warmup lr')
    parser.add_argument('--final_lr', type=float, default=1e-4, help='final lr')
    parser.add_argument('--lr_F', type=float, default=1e-3, help='learning rate of F')
    parser.add_argument('--iter_per_epoch', type=int, default=1, help='iter per epoch')
    parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')

    parser.add_argument('--alpha', type=float, default=1, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta', type=float, default=1, help='loss = l_data + alpha * l_PDE + beta * l_physics')

    parser.add_argument('--save_folder', type=str, default=None, help='save folder')
    parser.add_argument('--log_dir', type=str, default=None, help='log dir, if None, do not save')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    mlp = MLP()
    cnn = CNN()
    kan_small = KAN_small()
    kan_large = kan()
    kan_medium = KAN_medium()
    transformer = Transformer()
    print('cnn:')
    count_parameters(cnn)
    print('mlp:')
    count_parameters(mlp)    
    print('kan_small:')
    count_parameters(kan_small)
    print('kan_medium:')
    count_parameters(kan_medium)
    print('kan_large:')
    count_parameters(kan_large)
    print('transformer:')
    count_parameters(transformer)

    pinn = PINN_opt(args)
    print('pinn:')
    count_parameters(pinn.solution_u)
    count_parameters(pinn.dynamical_F)
    pikan_hgs = PIKAN_hgs(args)
    print('pikan_medium:')
    count_parameters(pikan_hgs.solution_u)
    count_parameters(pikan_hgs.dynamical_F)
    pikan_s = PIKAN_small(args)
    print('pikan_small:')
    count_parameters(pikan_s.solution_u)
    count_parameters(pikan_s.dynamical_F)
    pikan = PIKAN_opt(args)
    print('pikan_large:')
    count_parameters(pikan.solution_u)
    count_parameters(pikan.dynamical_F)    