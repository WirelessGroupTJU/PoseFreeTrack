import os, argparse, torch, time
from mtools import check_dir, read_file, str_insert, str2bool
from multiprocessing import Pool

parser = argparse.ArgumentParser(description='Arguments parser for ACGAN training')
parser.add_argument('-p','--path', type=str, default='run')
parser.add_argument('-n','--name', type=str, default=None)
parser.add_argument('-num',  type=int, default=2)
parser.add_argument('-test', type=int, default=0)
parser.add_argument('-log',  type=str2bool,       default=True,     help='write log file')
parser.add_argument('-dev',  type=int, nargs='+', default=None,     help='default: all gpu; -1:do not use dev')
args = parser.parse_args()
dev_cnt = torch.cuda.device_count()
if args.dev is None:
    args.dev = list(range(dev_cnt))

print('>> %s\n' % str(args))
if args.path is not None:
    filename = os.path.join(args.path, args.name)
else:
    filename = args.name
def is_cmd(cmd):
    return (not cmd.startswith('#')) and (not cmd.startswith('cd')) and len(cmd)
timestamp = time.strftime('%y-%m-%d-%H-%M-%S', time.localtime(time.time()))
cmds = read_file('%s.sh'%filename)
cmds = list(filter(is_cmd, cmds))
check_dir(filename)

def long_time_task(i, cmd):
    if len(cmd):
        param = ''
        if args.dev[0] != -1:
            param = '-dev %d' % (args.dev[int(i)%len(args.dev)])
        if args.log:
            param += ' > %s.log'%(os.path.join(filename, '%s_%d'%(timestamp,i)))
        if '#' in cmd:
            cmd = cmd.replace('#', ' %s #'%param, 1)
        else:
            cmd = '%s %s'%(cmd, param)
        print('Run task %s (%s): %s' % (i, os.getpid(), cmd))
        if args.test==0:
            os.system(cmd)

if __name__=='__main__':
    p = Pool(args.num)
    for i,cmd in zip(range(len(cmds)),cmds):
        p.apply_async(long_time_task, args=(i,cmd,))
    p.close()
    p.join()