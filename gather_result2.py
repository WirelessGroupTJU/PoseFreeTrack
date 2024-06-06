from mtools import load_json, join_path, csvread, write_file, save_json, check_dir
import numpy as np
import os
import argparse
import ipdb as pdb

def __main__():
    print(args)
    results_name = join_path('results', '%s_results.json'%args.job)
    results = load_json(results_name) if os.path.exists(results_name) else {}
    row_num = 0
    for output in args.outputs:
        data_path = join_path('output', args.job, output, '%s_1_N'%args.data_name)
        if os.path.exists(data_path):
            all_filenames = os.listdir(data_path)
            result_names = sorted(list(filter(lambda x: True if x.startswith(args.file_prefix) and x.endswith('.csv') else False, all_filenames)))            
            if len(result_names):
                errs = []
                for filename in result_names:
                    data = csvread(join_path(data_path, filename))
                    errs.append(data)
                errs = np.stack(errs)
                mean_err = np.round(np.mean(errs, 0), 4)
                if mean_err.shape[0]>row_num:
                    row_num = mean_err.shape[0]
                results[output] = {'node':args.node, 'mean_err':mean_err.tolist(), 'errs':errs.tolist()}
    csv_results = []
    for i in range(row_num):
        csv_results.extend(['%s,%d,%s'%(_, i, ','.join([str(__) for __ in results[_]['mean_err'][i]])) for _ in results])
    write_file(join_path('results', '%s_results.csv'%args.job), csv_results)
    save_json(results_name, results)
    print(results)
    print(results.keys())
    print(csv_results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--node', type=str)
    parser.add_argument('-j','--job', type=str, default='transposemt2')
    parser.add_argument('-dn','--data_name', type=str, default='oxiod_h5', choices=['oxiod', 'oxiod_h5', 'ridi', 'tjuimu'])
    parser.add_argument('-o','--outputs', type=str, default=None, nargs='+')
    parser.add_argument('--file_prefix', type=str, default='transposemt3_eval_result_loss_e') # output/transposemt/tpbase/oxiod_1_N/
    args = parser.parse_args()
    __main__()