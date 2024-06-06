import ipdb as pdb
import argparse
from basictorch_v2.trainer import get_optim
from basictorch_v2.tools import set_args_config, set_device, seed_everything, seed_numpy
from mtools import str2bool
from data.dataset import get_dataset, get_list, Datasets
from data.tools import eval_targets, set_args, extract_features
from models import TransPoseMT, TransPoseMT3, LIONet, RoNIN, TPTrainer

model_dict = {
    'transposemt': TransPoseMT,
    'transposemt3': TransPoseMT3,
    'ionet': LIONet,
    'ronin': RoNIN,
}

# Parsing arguments
parser = argparse.ArgumentParser(description='paramters for training transpose')
parser.add_argument('-c','--config',        type=str,           default=None,           help='config filenames', nargs='+')
parser.add_argument('-m','--model',         type=str,           default='transposemt')
parser.add_argument('-z','--proj_size',     type=int,           default=8)
parser.add_argument('-hs','--hidden_size',  type=int,           default=96)

parser.add_argument('-et','--enc_type',     type=str,           default='lstm',     choices=['lstm','gru','resnet'])
parser.add_argument('-gt','--gen_type',     type=str,           default='lstm',     choices=['lstm','gru','resnet'])
parser.add_argument('-det','--dec_type',    type=str,           default='lstm',     choices=['lstm','gru','resnet'])
parser.add_argument('-dt','--dis_type',     type=str,           default='lstm',     choices=['lstm','gru','resnet'])
parser.add_argument('-pt','--pred_type',    type=str,           default='lstm',     choices=['lstm','gru','resnet'])

parser.add_argument('-fl','--is_flat',      type=str2bool,      default=False) # for TransPoseMT3, all flat
parser.add_argument('-ef','--enc_flat',     type=str2bool,      default=False)
parser.add_argument('-def','--dec_flat',    type=str2bool,      default=True)
parser.add_argument('-gf','--gen_flat',     type=str2bool,      default=False)
parser.add_argument('-dif','--dis_flat',    type=str2bool,      default=True)
parser.add_argument('-pf','--pred_flat',    type=str2bool,      default=False)

parser.add_argument('-ebi','--enc_bidirect', type=str2bool,     default=False)
parser.add_argument('-debi','--dec_bidirect',type=str2bool,     default=True)
parser.add_argument('-gbi','--gen_bidirect', type=str2bool,     default=False)
parser.add_argument('-dibi','--dis_bidirect',type=str2bool,     default=True)
parser.add_argument('-pbi','--pred_bidirect',type=str2bool,     default=True)

parser.add_argument('-dl','--dis_linear',   type=str2bool,      default=False)
parser.add_argument('-pl','--pred_linear',  type=str2bool,      default=False)

parser.add_argument('-efu','--enc_fuse',    type=str2bool,      default=False) #TransPoseMT3

parser.add_argument('-act','--activation',  type=str,           default=None)
parser.add_argument('-ce','--cond_enc',     type=str2bool,      default=False)
parser.add_argument('-el','--embedding_label',type=str2bool,    default=False)
parser.add_argument('-d','--dropout',       type=float,         default=0.05)
parser.add_argument('-nl','--num_layers',   type=int,           default=[1,1,1], nargs='+')
parser.add_argument('-adv','--adv_loss',    type=str,           default='js',    choices=['js', 'w', 'ls', 'jse'])

parser.add_argument('--a_ae',               type=float,         default=1)
parser.add_argument('--a_grl',              type=float,         default=1)
parser.add_argument('--a_gan',              type=float,         default=1)
parser.add_argument('--a_cycx',             type=float,         default=1)
parser.add_argument('--a_cycz',             type=float,         default=1)
parser.add_argument('--a_cycsz',            type=float,         default=1)
parser.add_argument('--t_cycx',             type=str,           default='l1')
parser.add_argument('--t_cycz',             type=str,           default='fm')
parser.add_argument('--t_cycsz',            type=str,           default='gram')
parser.add_argument('--a_pred',             type=float,         default=10)
parser.add_argument('--a_pred_st',          type=float,         default=1)
parser.add_argument('--t_s_random',         type=str2bool,      default=False)

# # data arguments 
parser.add_argument('-n','--data_name',     type=str,           default='oxiod_h5', choices=['oxiod', 'oxiod_h5', 'ridi', 'tjuimu'])
parser.add_argument('-v','--data_ver',      type=str,           default='1',        help='data version')
parser.add_argument('-fm','--feature_mode', type=str,           default='N',        help='feature mode')
parser.add_argument('-lt','--label_type',   type=str,           default='polar',    choices=['polar', 'polarH', 'polarHW'])
parser.add_argument('-lm','--label_mode',   type=str,           default='sparse',   choices=['sparse', 'dense', 'mix'])
parser.add_argument('-o','--output',        type=str,           default='tmp',      help='output, the folder name of output')
parser.add_argument('-sp','--source_pose',  type=int,           default=0)
parser.add_argument('--val_split',          type=float,         default=0.1)
parser.add_argument('-du','--data_usage',   type=float,         default=1.0)
parser.add_argument('--window_size',        type=int,           default=200)
parser.add_argument('--step_size',          type=int,           default=10)
parser.add_argument('--dense_step_size',    type=int,           default=50)
parser.add_argument('--angle_stride',       type=int,           default=30)
parser.add_argument('--angle_window',       type=int,           default=10)
parser.add_argument('--feature_sigma',      type=float,         default=0.00001) # 0.00001
parser.add_argument('--label_sigma',        type=float,         default=0)
parser.add_argument('--noise_sigma',        type=float,         default=0.001)
parser.add_argument('-df', '--data_freq',   type=int,           default=100)
parser.add_argument('-am', '--acc_magn',    type=str2bool,      default=False)
parser.add_argument('-yd', '--yaw_diff',    type=str2bool,      default=False)
parser.add_argument('-ds', '--domain_std',  type=str2bool,      default=False)
parser.add_argument('-ydc', '--yaw_diff_cur',type=str2bool,     default=False)
parser.add_argument('-rt', '--rho_threshold',type=float,        default=2.1)

# # training arguments
parser.add_argument('-lf','--loss_func',    type=str,           default='mse',      choices=['mee', 'mse'])
parser.add_argument('-b','--batch_size',    type=int,           default=64,         help='batch size , default value=16')
parser.add_argument('-e','--epochs',        type=int,           default=50,         help='epochs, default value=100')
parser.add_argument('-ep','--epochs_pre',   type=int,           default=2)
parser.add_argument('--lr',                 type=float,         default=1e-04)
parser.add_argument('-st','--start_trail',  type=int,           default=0,          help='start trail time')
parser.add_argument('-t','--trails',        type=int,           default=1,          help='trail times')
parser.add_argument('-no','--exp_no',       type=str,           default='')
parser.add_argument('-dev','--device',      type=int,           default=0,          help='device')
parser.add_argument('-pb','--print_batch',  type=str2bool,      default=False,      help='print batch loss info')
parser.add_argument('--train_stage',        type=str2bool,      default=[True, True], nargs='+')
parser.add_argument('-load','--load_model', type=str2bool,      default=[False, False], nargs='+')
parser.add_argument('-i','--run_i',         type=str,           default='1')
parser.add_argument('-seed',                type=int,           default=0)
parser.add_argument('-nseed',               type=int,           default=None)
parser.add_argument('-exf','--extract_features', type=str2bool, default=False)
parser.add_argument('-moni','--monitor',    type=str,           default='t_mse')

args = set_args_config(parser)
torch_device = set_device(args)
set_args(args)
if args.seed is not None:
    seed_everything(args.seed)
elif args.nseed is not None:
    seed_numpy(args.seed)

pose_list, train_data_list, test_data_list = get_list(args.pose_list), get_list(args.data_list), get_list(args.test_data_list)
target_pose_list = pose_list.copy()
target_pose_list.pop(args.source_pose)

test_dataset = get_dataset(args.root_dir, test_data_list, pose_list, args, mode='test', label_type=args.label_type, target_pose_list=target_pose_list)
if not args.load_model[1]:
    train_dataset, val_dataset = get_dataset(args.root_dir, train_data_list, pose_list, args, mode='train', label_type=args.label_type, target_pose_list=target_pose_list)
    Ds = Datasets(True, True, args.val_split, args.batch_size, pose_list, train_dataset, test_dataset, val_dataset)

hyperparams = {
    'enc_type':args.enc_type, 
    'gen_type':args.gen_type, 
    'dec_type':args.dec_type, 
    'dis_type':args.dis_type,
    'pred_type':args.pred_type,

    'is_flat':args.is_flat, 
    'enc_flat':args.enc_flat, 
    'dec_flat':args.dec_flat, 
    'gen_flat':args.gen_flat, 
    'dis_flat':args.dis_flat,
    'pred_flat':args.pred_flat,

    'enc_bidirect':args.enc_bidirect, 
    'dec_bidirect':args.dec_bidirect, 
    'gen_bidirect':args.gen_bidirect, 
    'dis_bidirect':args.dis_bidirect,
    'pred_bidirect':args.pred_bidirect,

    'dis_linear':args.dis_linear,
    'pred_linear':args.pred_linear,

    'enc_fuse': args.enc_fuse,

    'activation':args.activation,
    'cond_enc':args.cond_enc,
    'embedding_label':args.embedding_label,
    'noise_sigma':args.noise_sigma,
    'step_size':args.dense_step_size,
    'window_size':args.window_size,
    'acc_magn':args.acc_magn,
    'yaw_diff':args.yaw_diff,

    'ae': args.a_ae,
    'grl': args.a_grl,
    'cgan': args.a_gan,
    'cycx': args.a_cycx,
    'cycz': args.a_cycz,
    'cycsz': args.a_cycsz,
    'cycx_type': args.t_cycx,
    'cycz_type': args.t_cycz,
    'cycsz_type': args.t_cycsz,
    'pred': args.a_pred,
    'pred_st': args.a_pred_st,
    't_s_random': args.t_s_random
}

for e in range(args.start_trail, args.start_trail+args.trails):
    model = model_dict[args.model](args.model, hyperparams, len(pose_list), args.hidden_size, args.proj_size, args.num_layers, dropout=args.dropout, adv_loss=args.adv_loss, label_mode=args.label_mode, label_type=args.label_type, loss_func=args.loss_func)
    model = model.to(torch_device)
    print(model)
    trainer = TPTrainer('transpose_trainer', args)
    trainer.outM.set_exp_no(e)
    print(args)
    if args.train_stage[0] and not args.load_model[1]:
        if args.load_model[0]:
            trainer.load_model(model, postfix='pre')
        else:
            model.stage = 1 # train enc dec pred
            trainer.fit(model, get_optim(model.parameters(), 'Adam'), Ds, args.batch_size, args.epochs_pre, initialize=True, postfix='pre', eval_train=True, batch_size_eval=512)
    if args.train_stage[1]:
        if args.load_model[1]:
            trainer.load_model(model)
        else:
            model.stage = 2 # train gen_s/t dis_s/t
            trainer.fit(model, get_optim(model.parameters(), 'Adam'), Ds, args.batch_size, args.epochs, initialize=False, monitor=args.monitor, test_monitor=args.monitor, eval_train=True, batch_size_eval=512)
    eval_targets(model, test_dataset, trainer.outM)
    if args.extract_features:
        extract_features(model, test_dataset, trainer.outM)
