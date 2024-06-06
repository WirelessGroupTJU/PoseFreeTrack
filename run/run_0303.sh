## Compare
# python train.py -c mt -o tpfinal3/mtbase
# python train.py -c mt -o tpfinal3/mtbase_aug -am true -yd true -ds true

## Ablation Study
python train.py -c tp5 -o tpfinal5_2/base
# python train.py -c tp4 -o tpfinal4/flat         --is_flat true
# python train.py -c tp4 -o tpfinal4/noAccMStd    --acc_magn false --domain_std false
# python train.py -c tp4 -o tpfinal4/noStd        --domain_std false
# python train.py -c tp4 -o tpfinal4/noYawDiff    --yaw_diff false
# python train.py -c tp4 -o tpfinal4/noDiff       --yaw_diff_cur true
# python train.py -c tp4 -o tpfinal4/noRec        --a_ae 0
# python train.py -c tp4 -o tpfinal4/noGAN        --a_gan 0
# python train.py -c tp4 -o tpfinal4/noCycZ       --a_cycz 0
# python train.py -c tp4 -o tpfinal4/noCycX       --a_cycx 0
# python train.py -c tp4 -o tpfinal4/noCycZX      --a_cycz 0 --a_cycx 0

# python train.py -c tp4 -o tpfinal4/noPre            --train_stage f t
# python train.py -c tp4 -o tpfinal4/windowsize50     --window_size 50
# python train.py -c tp4 -o tpfinal4/windowsize100    --window_size 100
# python train.py -c tp4 -o tpfinal4/windowsize300    --window_size 300
# python train.py -c tp4 -o tpfinal4/windowsize400    --window_size 400
# python train.py -c tp4 -o tpfinal4/a_ae/0.1     --a_ae 0.1
# python train.py -c tp4 -o tpfinal4/a_ae/10      --a_ae 10
# python train.py -c tp4 -o tpfinal4/a_gan/0.1    --a_gan 0.1
# python train.py -c tp4 -o tpfinal4/a_gan/10     --a_gan 10
# python train.py -c tp4 -o tpfinal4/a_cycx/0.1   --a_cycx 0.1
# python train.py -c tp4 -o tpfinal4/a_cycx/10    --a_cycx 10
# python train.py -c tp4 -o tpfinal4/a_cycz/0.1   --a_cycx 0.1
# python train.py -c tp4 -o tpfinal4/a_cycz/10    --a_cycx 10
# python train.py -c tp4 -o tpfinal4/a_pred/1     --a_pred 1
# python train.py -c tp4 -o tpfinal4/a_pred/100   --a_pred 100

# python make_cmd.py run/run_0303.sh -pn seed -pl 0 537 2024
# python gather_result2.py -n tjubd12 -j tpfinal3 -o base flat noAccMStd noStd noYawDiff noDiff noRec noGAN noCycZ noCycX noCycZX noPre windowsize50 windowsize100 windowsize300 windowsize400 a_ae/0.1 a_ae/10 a_gan/0.1 a_gan/10 a_cycx/0.1 a_cycx/10 a_cycz/0.1 a_cycz/10 a_pred/1 a_pred/100
# python gather_result2.py -n tjubd12 -j tpfinal4 -o base flat noAccMStd noStd noYawDiff noDiff noRec noGAN noCycZ noCycX noCycZX noPre windowsize50 windowsize100 windowsize300 windowsize400 a_ae/0.1 a_ae/10 a_gan/0.1 a_gan/10 a_cycx/0.1 a_cycx/10 a_cycz/0.1 a_cycz/10 a_pred/1 a_pred/100
# python gather_result2.py -n tjubd12 -j tpfinal5 -o base flat noAccMStd noStd noYawDiff noDiff noRec noGAN noCycZ noCycX noCycZX noPre windowsize50 windowsize100 windowsize300 windowsize400 a_ae/0.1 a_ae/10 a_gan/0.1 a_gan/10 a_cycx/0.1 a_cycx/10 a_cycz/0.1 a_cycz/10 a_pred/1 a_pred/100
# python gather_result2.py -n tjubd12 -j tpfinal5_2 -o base flat noAccMStd noStd noYawDiff noDiff noRec noGAN noCycZ noCycX noCycZX noCyc noPre windowsize50 windowsize100 windowsize300 windowsize400 a_ae/1 a_ae/100 a_gan/0.1 a_gan/10 a_cycx/0.1 a_cycx/10 a_cycz/0.1 a_cycz/10 a_pred/1 a_pred/100
# python gather_result2.py -n tjubd12 -j tpfinal5_2 --file_prefix transposemt_eval_result_loss_e -o mtbase mtbase_aug
