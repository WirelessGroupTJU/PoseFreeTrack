# python train.py -seed 0 -t 1 -c mt -o transposemtfinal/mt_base
# python train.py -seed 0 -t 1 -c tp -o transposemtfinal/base
# python train.py -seed 0 -t 1 -c tp -o transposemtfinal/acc_magn   -am true
# python train.py -seed 0 -t 1 -c tp -o transposemtfinal/yaw_diff   -yd true # yaw_diff_cur = rerange_angle(self.oris[seq_id][frame_id - self.window_size:frame_id, [-1]] - yaw_init_pre)
# python train.py -seed 0 -t 1 -c tp -o transposemtfinal/yaw_diff2   -yd true -yds false # yaw_diff_pre = rerange_angle(self.oris[seq_id][frame_id - self.window_size:frame_id, [-1]] - yaw_init_cur) + rerange_angle_value(yaw_init_cur-yaw_init_pre)
# python train.py -seed 0 -t 1 -c tp -o transposemtfinal/yaw_diff3   -yd true # yaw_diff_pre = rerange_angle(self.oris[seq_id][frame_id - self.window_size:frame_id, [-1]] - self.oris[seq_id][frame_id - self.window_size*2:frame_id- self.window_size, [-1]])
# python train.py -seed 0 -t 1 -c tp -o transposemtfinal/yaw_diff_sin   -yd true # yaw_diff_sin
# python train.py -seed 0 -t 1 -c tp -o transposemtfinal/acc_yaw -am true -yd true
# python train.py -seed 0 -t 1 -c tp -o transposemtfinal/domain_std  -ds true # acc_std, accm_std, gys_std
# python train.py -seed 0 -t 1 -c tp -o transposemtfinal/domain_std2 -ds true # acc_std, accm_std
# python train.py -seed 0 -t 1 -c tp -o transposemtfinal/accm_std -am true -ds true # accm_std
# python train.py -seed 0 -t 1 -c tp2 -o transposemtfinal/base_all # domain_std yaw_diff acc_magn
# python train.py -seed 0 -t 1 -c tp2 -o transposemtfinal/base2 # domain_std2
# python train.py -seed 0 -t 1 -c tp2 -o transposemtfinal/base3 # accm_std yaw_diff
# python train.py -seed 0 -t 1 -c tp2 -o transposemtfinal/base4 # accm_std yaw_diff2
# python train.py -seed 0 -t 1 -c tp2 -o transposemtfinal/base_yaw_diff3 # accm_std yaw_diff3
# python train.py -seed 0 -t 1 -c tp2 -o transposemtfinal/base_sin -yds true # accm_std yaw_diff_sin
# python train.py -seed 0 -t 1 -c tp2 -o transposemtfinal/base2_mix --label_mode mix

# python gather_result2.py -n tjubd5 -j transposemtfinal -o mt_base base acc_magn yaw_diff acc_yaw domain_std domain_std2 base_all base2 base2_mix accm_std base3 yaw_diff2 base4 yaw_diff3 yaw_diff_sin base_yaw_diff3 base_sin

# tpfinal2 add rho_threshold=2.1
# python train.py -seed 0 -t 1 -c tp  -o tpfinal2/tp_base2
# python train.py -seed 0 -t 1 -c tp  -o tpfinal2/accm_std -am true -ds true
# python train.py -seed 0 -t 1 -c tp  -o tpfinal2/yaw_diff -yd true # yaw_diff_pre = rerange_angle(self.oris[seq_id][frame_id - self.window_size:frame_id, [-1]] - self.oris[seq_id][frame_id - self.window_size*2:frame_id- self.window_size, [-1]])
# python train.py -seed 0 -t 1 -c tp  -o tpfinal2/yaw_diff -yd true
# python train.py -seed 0 -t 1 -c tp  -o tpfinal2/yaw_diff_pre -yd true # only pre, no cur
# python train.py -seed 0 -t 1 -c tp2 -o tpfinal2/tpfinal2
# python train.py -seed 0 -t 1 -c tp3 -o tpfinal3/tpfinal3 # gram deep, yaw_diff_pre
# python gather_result2.py -n tjubd5 -j tpfinal2 -o tp_base2 accm_std yaw_diff tpfinal2 yaw_diff_pre tpfinal3

# compare
# python train.py -seed 0 -st 0 -t 1 -c tp  -o tpfinal2/mt_base
# # python train.py -seed 0 -st 0 -t 1 -c tp2 -o tpfinal2/tpfinal2

# python train.py -seed 537 -st 1 -t 1 -c tp  -o tpfinal2/mt_base
# python train.py -seed 537 -st 1 -t 1 -c tp2 -o tpfinal2/tpfinal2

# python train.py -seed 2024 -st 2 -t 1 -c tp  -o tpfinal2/mt_base
# python train.py -seed 2024 -st 2 -t 1 -c tp2 -o tpfinal2/tpfinal2

# python train.py -seed 0    -ep 0 -st 0 -t 1 -c tp2 -o tpfinal2/nopre -dev 1
# python train.py -seed 537  -ep 0 -st 1 -t 1 -c tp2 -o tpfinal2/nopre -dev 1
# python train.py -seed 2024 -ep 0 -st 2 -t 1 -c tp2 -o tpfinal2/nopre -dev 1

# python gather_result2.py -n tjubd5 -j tpfinal2 -o mt_base tpfinal2 nopre

python train.py -c tp3 -o tpfinal3/tpfinal3 # --train_stage f t --load_model f t --extract_features t
python train.py -c tp3 -o tpfinal3_ridi/tpfinal3 -n ridi
python train.py -c tp3 -o tpfinal3_tjuimu/tpfinal3 -n tjuimu
python train.py -c tp3 -o tpfinal3/flat --is_flat true

python train.py -seed 0 -t 1 -c mt -o transposemtfinal/mt_base --train_stage f t --load_model f t --extract_features t
# python train.py -c tp3 -o tpfinal3/noAccMStd   --acc_magn false --domain_std false
# python train.py -c tp3 -o tpfinal3/noStd       --domain_std false
# python train.py -c tp3 -o tpfinal3/noYawDiff   --yaw_diff false
# python train.py -c tp3 -o tpfinal3/noDiff      --yaw_diff_cur true
# python train.py -c tp3 -o tpfinal3/noRec       --a_ae 0
# python train.py -c tp3 -o tpfinal3/noGAN       --a_gan 0
# python train.py -c tp3 -o tpfinal3/noCycZ      --a_cycz 0
# python train.py -c tp3 -o tpfinal3/noCycX      --a_cycx 0

# python gather_result2.py -n tjubd5 -j tpfinal3 -o tpfinal3 flat

# --train_stage f t --load_model f t
# python train.py --train_stage f t --load_model f t -c tp3 -o tpfinal3/noCycZX      --a_cycz 0 --a_cycx 0 -seed 0 -st 0
# python train.py --train_stage f t --load_model f t -c tp3 -o tpfinal3/noCycZX      --a_cycz 0 --a_cycx 0 -seed 537 -st 1
# python train.py --train_stage f t --load_model f t -c tp3 -o tpfinal3/noCycZX      --a_cycz 0 --a_cycx 0 -seed 2024 -st 2

# tpfinal4 add zero_thres=0.1
python train.py -c tp3 -o tpfinal4/base
python train.py -c tp3 -o tpfinal4/base_efu -efu True