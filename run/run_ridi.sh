# RIDI
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_2            -lt pos     -lf mee -lm mixin                      # tjubd5
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_noshal       -lt pos     -lf mee -lm mixin --a_cycsz 0          # stone
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_nodeep       -lt pos     -lf mee -lm mixin --a_cycz 0           # stone
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_fmshal       -lt pos     -lf mee -lm mixin --t_cycsz fm         # stone
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_gramdeep     -lt pos     -lf mee -lm mixin --t_cycz gram        # stone
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_nocycx       -lt pos     -lf mee -lm mixin --a_cycx 0           # stone
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_flatenc      -lt pos     -lf mee -lm mixin --enc_flat t         # stone
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_plainenc     -lt pos     -lf mee -lm mixin --enc_dilation 1     # stone
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_polar        -lt polar           -lm mixin                      # stone
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_sparse       -lt pos     -lf mee -lm sparse                     # stone
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_nopre        -lt pos     -lf mee -lm mixin -ep 0 0              # stone
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_seq100       -lt pos     -lf mee -lm mixin --window_size 100    # tjubd3
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_seq300       -lt pos     -lf mee -lm mixin --window_size 300    # tjubd3
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_seq400       -lt pos     -lf mee -lm mixin --window_size 400    # tjubd3
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_cycsz0.1     -lt pos -lf mee -lm mixin --a_cycsz 0.1            # tjubd3
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_cycz0.1      -lt pos -lf mee -lm mixin --a_cycz  0.1            # tjubd3
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_cycx0.1      -lt pos -lf mee -lm mixin --a_cycx  0.1            # tjubd3
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_pred1        -lt pos -lf mee -lm mixin --a_pred  1              # tjubd3
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_gan0.1       -lt pos -lf mee -lm mixin --a_gan   0.1            # tjubd3
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_cycsz10      -lt pos -lf mee -lm mixin --a_cycsz 10             # tjudb5
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_cycz10       -lt pos -lf mee -lm mixin --a_cycz  10             # tjudb5
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_cycx10       -lt pos -lf mee -lm mixin --a_cycx  10             # tjudb5
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_pred100      -lt pos -lf mee -lm mixin --a_pred  100            # tjudb5
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_gan10        -lt pos -lf mee -lm mixin --a_gan   10             # tjudb5
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_cycsz0.01    -lt pos -lf mee -lm mixin --a_cycsz 0.01           # tjudb5
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_cycsz100     -lt pos -lf mee -lm mixin --a_cycsz 100            # tjudb5
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_cycz0.01     -lt pos -lf mee -lm mixin --a_cycz 0.01            # tjudb5
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_cycz100      -lt pos -lf mee -lm mixin --a_cycz 100             # tjudb5
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_cycx0.01     -lt pos -lf mee -lm mixin --a_cycx 0.01            # tjudb12
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_cycx100      -lt pos -lf mee -lm mixin --a_cycx 100             # tjudb12
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_pred0.1      -lt pos -lf mee -lm mixin --a_pred 0.1             # tjudb12
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_pred1000     -lt pos -lf mee -lm mixin --a_pred 1000            # tjudb12
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_gan0.01      -lt pos -lf mee -lm mixin --a_gan 0.01             # tjudb12
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_gan100       -lt pos -lf mee -lm mixin --a_gan 100              # tjudb12

