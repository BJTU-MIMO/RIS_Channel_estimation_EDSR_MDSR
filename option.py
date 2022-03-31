import argparse
import template

parser = argparse.ArgumentParser(description='EDSR and MDSR')
# 模型
parser.add_argument('--model', default='EDSR',
                    help='model name')
parser.add_argument('--save', type=str, default='EDSR',
                    help='file name to save')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# 参数
parser.add_argument('--scale', default='2',
                    help='super resolution scale')
parser.add_argument('--epochs', type=int, default=400,
                    help='number of epochs to train')
parser.add_argument('--n_resblocks', type=int, default=12,
                    help='number of residual blocks')
parser.add_argument('--print_every', type=int, default=50,
                    help='how many batches to wait before logging training status')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--n_colors', type=int, default=1,
                    help='number of color channels to use')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--batch_size', type=int, default=10,
                    help='input batch size for training')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=200,
                    help='learning rate decay per N epochs')

parser.add_argument('--img_height', type=int, default=256,
                    help='image height number')
parser.add_argument('--img_width', type=int, default=32,
                    help='image width number')
parser.add_argument('--img_sr_height', type=int, default=64,
                    help='image sr number')
parser.add_argument('--img_reshape', type=int, default=128,
                    help='image reshape height')

parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')


parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--load', type=str, default='.',
                    help='file name to load')

parser.add_argument('--degradation', type=str, default='BI',
                    help='degradation model: BI, BD')

parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--testset', type=str, default='Set5',
                    help='dataset name for testing')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')

parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')

parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')


parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')

parser.add_argument('--pre_train', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')

parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')


parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--print_model', action='store_true',
                    help='print model')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e6',
                    help='skipping batch that has large error')



args = parser.parse_args()
template.set_template(args)

args.scale = list(map(lambda x: int(x), args.scale.split('+')))

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
