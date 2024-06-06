# python train.py -c tp5 -o tpfinal5_3/a_ae/1     --a_ae 1
# python train.py -c tp5 -o tpfinal5_3/a_ae/100     --a_ae 100
# python train.py -c tp5 -o tpfinal5_3/a_gan/0.1    --a_gan 0.1
# python train.py -c tp5 -o tpfinal5_3/a_gan/10     --a_gan 10
# python train.py -c tp5 -o tpfinal5_3/a_cycx/0.1   --a_cycx 0.1
# python train.py -c tp5 -o tpfinal5_3/a_cycx/10    --a_cycx 10
# python train.py -c tp5 -o tpfinal5_3/a_cycz/0.1   --a_cycz 0.1
# python train.py -c tp5 -o tpfinal5_3/a_cycz/10    --a_cycz 10
# python train.py -c tp5 -o tpfinal5_3/a_pred/1     --a_pred 0.1
# python train.py -c tp5 -o tpfinal5_3/a_pred/100   --a_pred 100

# python train.py -c tp5 -o tpfinal5_3/a_ae/1     --a_ae 1 -seed 537 -st 1
# python train.py -c tp5 -o tpfinal5_3/a_ae/100     --a_ae 100 -seed 537 -st 1
# python train.py -c tp5 -o tpfinal5_3/a_gan/0.1    --a_gan 0.1 -seed 537 -st 1
# python train.py -c tp5 -o tpfinal5_3/a_gan/10     --a_gan 10 -seed 537 -st 1
# python train.py -c tp5 -o tpfinal5_3/a_cycx/0.1   --a_cycx 0.1 -seed 537 -st 1
# python train.py -c tp5 -o tpfinal5_3/a_cycx/10    --a_cycx 10 -seed 537 -st 1
# python train.py -c tp5 -o tpfinal5_3/a_cycz/0.1   --a_cycz 0.1 -seed 537 -st 1
# python train.py -c tp5 -o tpfinal5_3/a_cycz/10    --a_cycz 10 -seed 537 -st 1
# python train.py -c tp5 -o tpfinal5_3/a_pred/1     --a_pred 1 -seed 537 -st 1
# python train.py -c tp5 -o tpfinal5_3/a_pred/100   --a_pred 100 -seed 537 -st 1

# python train.py -c tp5 -o tpfinal5_3/a_ae/1     --a_ae 1 -seed 2024 -st 2
# python train.py -c tp5 -o tpfinal5_3/a_ae/100     --a_ae 100 -seed 2024 -st 2
# python train.py -c tp5 -o tpfinal5_3/a_gan/0.1    --a_gan 0.1 -seed 2024 -st 2
# python train.py -c tp5 -o tpfinal5_3/a_gan/10     --a_gan 10 -seed 2024 -st 2
# python train.py -c tp5 -o tpfinal5_3/a_cycx/0.1   --a_cycx 0.1 -seed 2024 -st 2
# python train.py -c tp5 -o tpfinal5_3/a_cycx/10    --a_cycx 10 -seed 2024 -st 2
# python train.py -c tp5 -o tpfinal5_3/a_cycz/0.1   --a_cycz 0.1 -seed 2024 -st 2
# python train.py -c tp5 -o tpfinal5_3/a_cycz/10    --a_cycz 10 -seed 2024 -st 2
# python train.py -c tp5 -o tpfinal5_3/a_pred/1     --a_pred 1 -seed 2024 -st 2
# python train.py -c tp5 -o tpfinal5_3/a_pred/100   --a_pred 100 -seed 2024 -st 2

python train.py -c tp5 -o tpfinal5_3/base -seed 0 -st 0
# python train.py -c tp5 -o tpfinal5_3/noAccMStd    --acc_magn false --domain_std false -seed 0 -st 0
# python train.py -c tp5 -o tpfinal5_3/noStd        --domain_std false -seed 0 -st 0
# python train.py -c tp5 -o tpfinal5_3/noYawDiff    --yaw_diff false -seed 0 -st 0
# python train.py -c tp5 -o tpfinal5_3/noDiff       --yaw_diff_cur true -seed 0 -st 0
# python train.py -c tp5 -o tpfinal5_3/noRec        --a_ae 0 -seed 0 -st 0
# python train.py -c tp5 -o tpfinal5_3/noGAN        --a_gan 0 -seed 0 -st 0
# python train.py -c tp5 -o tpfinal5_3/noCycZ       --a_cycz 0 -seed 0 -st 0
# python train.py -c tp5 -o tpfinal5_3/noCycX       --a_cycx 0 -seed 0 -st 0
# python train.py -c tp5 -o tpfinal5_3/noCycZX      --a_cycz 0 --a_cycx 0 -seed 0 -st 0

python train.py -c tp5 -o tpfinal5_3/base -seed 537 -st 1
# python train.py -c tp5 -o tpfinal5_3/noAccMStd    --acc_magn false --domain_std false -seed 537 -st 1
# python train.py -c tp5 -o tpfinal5_3/noStd        --domain_std false -seed 537 -st 1
# python train.py -c tp5 -o tpfinal5_3/noYawDiff    --yaw_diff false -seed 537 -st 1
# python train.py -c tp5 -o tpfinal5_3/noDiff       --yaw_diff_cur true -seed 537 -st 1
# python train.py -c tp5 -o tpfinal5_3/noRec        --a_ae 0 -seed 537 -st 1
# python train.py -c tp5 -o tpfinal5_3/noGAN        --a_gan 0 -seed 537 -st 1
# python train.py -c tp5 -o tpfinal5_3/noCycZ       --a_cycz 0 -seed 537 -st 1
# python train.py -c tp5 -o tpfinal5_3/noCycX       --a_cycx 0 -seed 537 -st 1
# python train.py -c tp5 -o tpfinal5_3/noCycZX      --a_cycz 0 --a_cycx 0 -seed 537 -st 1

python train.py -c tp5 -o tpfinal5_3/base -seed 2024 -st 2
# python train.py -c tp5 -o tpfinal5_3/noAccMStd    --acc_magn false --domain_std false -seed 2024 -st 2
# python train.py -c tp5 -o tpfinal5_3/noStd        --domain_std false -seed 2024 -st 2
# python train.py -c tp5 -o tpfinal5_3/noYawDiff    --yaw_diff false -seed 2024 -st 2
# python train.py -c tp5 -o tpfinal5_3/noDiff       --yaw_diff_cur true -seed 2024 -st 2
# python train.py -c tp5 -o tpfinal5_3/noRec        --a_ae 0 -seed 2024 -st 2
# python train.py -c tp5 -o tpfinal5_3/noGAN        --a_gan 0 -seed 2024 -st 2
# python train.py -c tp5 -o tpfinal5_3/noCycZ       --a_cycz 0 -seed 2024 -st 2
# python train.py -c tp5 -o tpfinal5_3/noCycX       --a_cycx 0 -seed 2024 -st 2
# python train.py -c tp5 -o tpfinal5_3/noCycZX      --a_cycz 0 --a_cycx 0 -seed 2024 -st 2

# python train.py -c tp5 -o tpfinal5_3/noPre            --train_stage f t
# python train.py -c tp5 -o tpfinal5_3/windowsize50     --window_size 50
# python train.py -c tp5 -o tpfinal5_3/windowsize100    --window_size 100
# python train.py -c tp5 -o tpfinal5_3/windowsize300    --window_size 300
# # python train.py -c tp5 -o tpfinal5_3/windowsize400    --window_size 400

# python train.py -c tp5 -o tpfinal5_3/noCyc --a_cycz 0 --a_cycx 0 --a_pred_st 0 -seed 0    -st 0
# python train.py -c tp5 -o tpfinal5_3/noCyc --a_cycz 0 --a_cycx 0 --a_pred_st 0 -seed 537  -st 1
# python train.py -c tp5 -o tpfinal5_3/noCyc --a_cycz 0 --a_cycx 0 --a_pred_st 0 -seed 2024 -st 2