import pandas as pd
import os.path as osp
import numpy as np
# from mtools.plot_preamble import *
from mtools import list_find, list_ind, read_file, save_json
import ipdb as pdb

# def load_data(file_path):
#     store = pd.HDFStore(file_path, mode='r')
#     all_df = store.get('all_imu')
#     store.close()
#     return all_df

# filepath_list = read_file('/mnt/lun2/home/wjk/workspace/DeepTrack-pr32/Configs/Data_List/oxiod_list_train.txt')
# filepath_list = list(filter(lambda x: not x.startswith('#'), filepath_list))
# domains = ['handheld', 'handbag', 'pocket', 'trolley']

# dfs = [load_data(osp.join('Data/OxIOD', file_path, 'data.h5')) for file_path in filepath_list]

# all_domains = [trip.split('/')[0] for trip in filepath_list] 
# domain_inds_list = [list_find(all_domains, domain) for domain in domains]

# print(domain_inds_list)
# domain_stats = {}

# for domain_inds, domain in zip(domain_inds_list, domains):
#     df = pd.concat(list_ind(dfs, domain_inds))
#     acc_np = df[['AccX', 'AccY', 'AccZ']].values.astype("float32")
#     df['AccM'] = np.linalg.norm(acc_np, axis=-1)
#     acc_mu   = np.mean(df[['AccX', 'AccY', 'AccZ']].values)
#     acc_sigma = np.std(df[['AccX', 'AccY', 'AccZ']].values-acc_mu)
#     print(acc_mu, acc_sigma)

#     accmagn_mu = np.mean(df['AccM'].values)
#     accmagn_sigma = np.std(df['AccM'].values)
#     print(accmagn_mu, accmagn_sigma)

#     gys_mu   = np.mean(df[['GysX', 'GysY', 'GysZ']].values)
#     gys_sigma = np.std(df[['GysX', 'GysY', 'GysZ']].values-gys_mu)
#     print(gys_mu, gys_sigma)

#     domain_stats[domain] = np.array([acc_mu, acc_sigma, gys_mu, gys_sigma, accmagn_mu, accmagn_sigma]).tolist()

# print(domain_stats)
# save_json(f'/mnt/lun2/home/wjk/workspace/DeepTrack-pr32/Configs/Data_List/oxiod_domain_stat.json', domain_stats)

# filepath_list = read_file('/mnt/lun2/home/wjk/workspace/DeepTrack/configs/lists/ridi_train_list.txt')
# filepath_list = list(filter(lambda x: not x.startswith('#'), filepath_list))
# all_domains = [trip.split(',')[1] for trip in filepath_list] 
# filepath_list = [trip.split(',')[0] for trip in filepath_list] 

# domains = ['handheld', 'body', 'leg', 'bag']
# domain_inds_list = [list_find(all_domains, domain) for domain in domains]
# print(all_domains)
# print(domain_inds_list)

# dfs = [pd.read_csv(osp.join('data/ridi', file_path, 'processed', 'data.csv')) for file_path in filepath_list]
# domain_stats = {}

# for domain_inds, domain in zip(domain_inds_list, domains):
#     df = pd.concat(list_ind(dfs, domain_inds))
#     df = df[::2]
#     gyro = df[['gyro_x', 'gyro_y', 'gyro_z']].values
#     acce = df[['acce_x', 'acce_y', 'acce_z']].values
#     grav = df[['grav_x', 'grav_y', 'grav_z']].values
#     acc_np = (acce - grav) / 9.8

#     df['AccM'] = np.linalg.norm(acc_np, axis=-1)

#     acc_mu   = np.mean(acc_np)
#     acc_sigma = np.std(acc_np-acc_mu)
#     print(acc_mu, acc_sigma)

#     accmagn_mu = np.mean(df['AccM'].values)
#     accmagn_sigma = np.std(df['AccM'].values)
#     print(accmagn_mu, accmagn_sigma)

#     gys_mu   = np.mean(df[['gyro_x', 'gyro_y', 'gyro_z']].values)
#     gys_sigma = np.std(df[['gyro_x', 'gyro_y', 'gyro_z']].values-gys_mu)
#     print(gys_mu, gys_sigma)

#     domain_stats[domain] = np.array([acc_mu, acc_sigma, gys_mu, gys_sigma, accmagn_mu, accmagn_sigma]).tolist()

#     # index_list = range(200*2, df.shape[0]-1, 10)
#     # starts = [np.linalg.norm(df.iloc[index-200][['pos_x', 'pos_y']].values - df.iloc[index][['pos_x', 'pos_y']].values) for index in index_list]
#     # rho_df = pd.DataFrame(np.stack([np.linalg.norm(df.iloc[index-200][['pos_x', 'pos_y']].values - df.iloc[index][['pos_x', 'pos_y']].values) for index in index_list], axis=0), columns=['rho'])
#     # print(rho_df.describe([0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
# # print(domain_stats)
# # save_json(f'/mnt/lun2/home/wjk/workspace/DeepTrack/data/data_conf/ridi_domain_stat.json', domain_stats)

#                 rho
# count  15784.000000
# mean       2.176283
# std        1.189001
# min        0.001168
# 5%         0.722253
# 25%        1.972885
# 50%        2.273104
# 75%        2.444017
# 90%        2.671140
# 95%        2.805946
# 99%        3.237649
# max       28.319495
#                 rho
# count  16221.000000
# mean       2.112568
# std        1.578423
# min        0.000616
# 5%         0.126702
# 25%        1.852167
# 50%        2.228909
# 75%        2.509347
# 90%        2.740815
# 95%        2.898431
# 99%        3.299490
# max       42.032191
#                 rho
# count  10785.000000
# mean       2.038813
# std        1.169799
# min        0.000522
# 5%         0.181265
# 25%        1.879988
# 50%        2.196365
# 75%        2.441116
# 90%        2.621546
# 95%        2.725262
# 99%        2.892414
# max       24.882583
#                 rho
# count  12215.000000
# mean       2.338178
# std        2.414338
# min        0.000062
# 5%         0.666197
# 25%        2.042965
# 50%        2.396913
# 75%        2.580410
# 90%        2.733292
# 95%        2.862126
# 99%        3.175673
# max       43.159469

filepath_list = read_file('/mnt/lun2/home/wjk/workspace/DeepTrack/configs/lists/tjuimu_train_list.txt')
filepath_list = list(filter(lambda x: not x.startswith('#'), filepath_list))
all_domains = [trip.split('#')[2] for trip in filepath_list]

domains = ['1', '2', '3', '4']
domain_inds_list = [list_find(all_domains, domain) for domain in domains]
print(all_domains)
print(domain_inds_list)

dfs = [pd.read_csv(osp.join('data/tjuimu', file_path, 'IMU_with_label_raw.txt'), delimiter=' ', header=None, names=['acce_x', 'acce_y', 'acce_z', 'gyro_x', 'gyro_y', 'gyro_z', 'pos_x', 'pos_y']) for file_path in filepath_list]
domain_stats = {}

for domain_inds, domain in zip(domain_inds_list, domains):
    df = pd.concat(list_ind(dfs, domain_inds))
    gys_np = np.deg2rad(df[['gyro_x', 'gyro_y', 'gyro_z']].values)
    acc_np = df[['acce_x', 'acce_y', 'acce_z']].values/9.8

    df['AccM'] = np.linalg.norm(acc_np, axis=-1)

    acc_mu   = np.mean(acc_np)
    acc_sigma = np.std(acc_np-acc_mu)
    print(acc_mu, acc_sigma)

    accmagn_mu = np.mean(df['AccM'].values)
    accmagn_sigma = np.std(df['AccM'].values)
    print(accmagn_mu, accmagn_sigma)

    gys_mu   = np.mean(gys_np)
    gys_sigma = np.std(gys_np-gys_mu)
    print(gys_mu, gys_sigma)

    domain_stats[domain] = np.array([acc_mu, acc_sigma, gys_mu, gys_sigma, accmagn_mu, accmagn_sigma]).tolist()

    # index_list = range(200*2, df.shape[0]-1, 10)
    # starts = [np.linalg.norm(df.iloc[index-200][['pos_x', 'pos_y']].values - df.iloc[index][['pos_x', 'pos_y']].values) for index in index_list]
    # rho_df = pd.DataFrame(np.stack([np.linalg.norm(df.iloc[index-200][['pos_x', 'pos_y']].values - df.iloc[index][['pos_x', 'pos_y']].values) for index in index_list], axis=0), columns=['rho'])
    # print(rho_df.describe([0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).T)

print(domain_stats)
save_json(f'/mnt/lun2/home/wjk/workspace/DeepTrack/data/data_conf/tjuimu_domain_stat.json', domain_stats)


#       count      mean      std       min        5%       25%       50%       75%       90%       95%        99%        max
# rho  8233.0  3.155671  6.63551  0.000063  0.389285  1.617578  2.255489  2.709739  2.914978  3.042969  41.514746  49.921445

#       count      mean       std       min        5%       25%       50%       75%       90%       95%        99%       max
# rho  8227.0  3.188382  6.748036  0.000082  0.954221  1.573989  2.255848  2.714504  2.921437  3.028903  42.439223  50.54291

#       count      mean       std      min        5%       25%       50%       75%       90%       95%        99%        max
# rho  9419.0  2.791727  6.351976  0.00001  0.361721  1.150988  1.814074  2.687416  2.946634  3.036253  42.439019  49.998617

#       count      mean       std       min        5%       25%       50%     75%       90%      95%        99%        max
# rho  8410.0  2.648344  5.942768  0.000092  0.102619  1.007519  2.008995  2.6602  2.910237  2.99882  41.679464  47.881667