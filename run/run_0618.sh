# python train.py -seed 0 -t 3 -c tp -lt polarHW -o transposemt3/weights/predw1.5 --a_pred_w 1.5 1 1
# python train.py -seed 0 -t 3 -c tp -lt polarHW -o transposemt3/weights/predw2   --a_pred_w 2 1 1
# python train.py -seed 0 -t 3 -c tp -lt polarHW -o transposemt3/weights/predw3   --a_pred_w 3 1 1
# python train.py -seed 0 -t 3 -c tp -lt polarHW -o transposemt3/weights/predw5   --a_pred_w 5 1 1
# python gather_result2.py -n tjubd12 -j transposemt3 -o weights/predw1.5 weights/predw2 weights/predw3

# python train.py -seed 0 -t 3 -c tp -lt polarHW -o transposemt3/tp_noCycsz --a_cycsz 0
python train.py -seed 0 -t 3 -c tp -lt polarHW -o transposemt3/tp_encflat --enc_flat true --a_cycsz 0
python train.py -seed 0 -t 3 -c tp -lt polarHW -o transposemt3/tp_condEnc --cond_enc true --a_cycsz 0

python train.py -seed 0 -t 1 -c tp -lt polarHW -o tmp --enc_flat true --enc_type resnet --a_cycsz 0
python train.py -seed 0 -t 1 -c tp -lt polarHW -o tmp --enc_type resnet --a_cycsz 0 -du 0.3
# python gather_result2.py -n tjubd12 -j transposemt3 -o tp_encflat tp_condEnc

# python train.py -seed 0 -t 3 -c mt -lt polarHW -o transposemt3/mt_base                   # tjubd12
# python train.py -seed 0 -t 3 -c tp -lt polarHW -o transposemt3/tp_base                   # tjubd12
# python train.py -seed 0 -t 3 -c mt -lt polarH -o transposemt3/mt_polarH                  # tjubd12
# python train.py -seed 0 -t 3 -c tp -lt polarH -o transposemt3/tp_polarH                  # tjubd12
# python train.py -seed 0 -t 3 -c tp -lt polarHW -o transposemt3/tp_base --a_cycsz 0          # stone
# python train.py -seed 0 -t 3 -c tp -lt polarHW -o transposemt3/tp_base --a_cycz 0           # stone
# python train.py -seed 0 -t 3 -c tp -lt polarHW -o transposemt3/tp_base --t_cycsz fm         # stone
# python train.py -seed 0 -t 3 -c tp -lt polarHW -o transposemt3/tp_base --t_cycz gram        # stone
# python train.py -seed 0 -t 3 -c tp -lt polarHW -o transposemt3/tp_base --a_cycx 0           # stone
# python train.py -seed 0 -t 3 -c tp -lt polarHW -o transposemt3/tp_base --enc_flat t         # stone
# python train.py -seed 0 -t 3 -c tp -lt polarHW -o transposemt3/tp_base --enc_dilation 1     # stone
# python gather_result2.py -n tjubd12 -o mt_base tp_base mt_polarH tp_polarH