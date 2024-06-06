from mtools import load_json, join_path, csvread, csvwrite, save_json, check_dir
import numpy as np
import os
import argparse

def get_outputs(filename):
    outputs = load_json(filename)['outputs_map'][args.localhost]
    print(args.localhost, outputs)
    return outputs

def get_outputs_paths(outputs):
    outputs_paths = [join_path('output', output) for output in outputs]
    print(outputs_paths)
    return  outputs_paths

def __main__():
    outputs = get_outputs(join_path(args.jid_prefix, args.jid+'.json'))
    outputs_paths = get_outputs_paths(outputs)

    results = []
    for output, outputs_path in zip(outputs, outputs_paths):
        result_names= list(filter(lambda x: True if x.startswith(args.file_prefix) else False, os.listdir(join_path(outputs_path, args.data_folder)))) if os.path.exists(join_path(outputs_path, args.data_folder)) else []
        errs = []
        for filename in result_names:
            errs.append(csvread(join_path(outputs_path, args.data_folder, filename)))
        mean_err = np.round(np.mean(errs, 0), 4) if len(errs) else np.zeros((4,))
        errs = np.stack(errs) if len(errs) else np.array([])
        results.append({'output':output, 'mean_err':mean_err.tolist(), 'errs':errs.tolist()})

    result = np.array([_['mean_err'] for _ in results])
    print(results)
    print(result)
    check_dir(join_path(args.jid_prefix, args.jid))
    csvwrite(join_path(args.jid_prefix, args.jid, args.localhost+'.csv'), result)
    save_json(join_path(args.jid_prefix, args.jid, args.localhost+'.json'), results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--localhost', type=str)
    parser.add_argument('-i', '--jid', type=str, default='run_tjuimu_2022-10-19-22-39-30')
    parser.add_argument('--jid_prefix', type=str, default='results')
    parser.add_argument('--data_folder', type=str, default='tjuimu_1_N')
    parser.add_argument('--file_prefix', type=str, default='transposemt_result_mee')
    args = parser.parse_args()
    __main__()