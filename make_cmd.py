import argparse
from mtools import read_file, write_file, print_each, load_cmds

parser = argparse.ArgumentParser(description='Arguments parser for ACGAN training')
parser.add_argument('path', type=str)
parser.add_argument('-pn', type=str, default=None)
parser.add_argument('--param_name', type=str, default=None)
parser.add_argument('-pl','--param_list', type=str, nargs='+', default=None)
args = parser.parse_args()

cmds = load_cmds(args.path)

params = args.param_list
param_name = f'--{args.param_name}' if args.param_name is not None else f'-{args.pn}'

new_cmds = []
for i,param in enumerate(params):
    _new_cmds = [f'{_} {param_name} {param} -st {i}' for _ in cmds]
    new_cmds.extend(_new_cmds)

print_each(cmds)
print_each(params)
print_each(new_cmds)

print(len(cmds), len(params), len(new_cmds))
write_file(args.path.replace('.sh', 's.sh'), new_cmds)
