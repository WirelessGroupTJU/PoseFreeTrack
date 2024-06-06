import argparse
import numpy as np
from mtools import join_path, load_json, list_con, save_json, csvwrite

def get_outputs(filename):
    outputs = load_json(filename)['outputs_map'][args.localhost]
    print(args.localhost, outputs)
    return outputs


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--jid', type=str, default='run_tjuimu_2022-10-19-22-39-30')
parser.add_argument('--jid_prefix', type=str, default='results')
args = parser.parse_args()

outputs_json = load_json(join_path(args.jid_prefix, args.jid+'.json'))
results = list_con([load_json(join_path('results', args.jid, ser+'.json')) for ser in outputs_json['servers']])
results_output = [_['output'] for _ in results]
results = [results[results_output.index(_)] for _ in outputs_json['outputs']]
result = np.array([_['mean_err'] for _ in results])
print(results)
print(result)
csvwrite(join_path(args.jid_prefix, args.jid+'.csv'), result)
save_json(join_path(args.jid_prefix, args.jid+'_results.json'), results)

# print(np.mean(result[:,1:], -1, keepdims=True))