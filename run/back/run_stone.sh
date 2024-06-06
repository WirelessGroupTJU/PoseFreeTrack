python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_noshal       -lt pos     -lf mee -lm mixin --a_cycsz 0          # stone
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_nodeep       -lt pos     -lf mee -lm mixin --a_cycz 0           # stone
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_fmshal       -lt pos     -lf mee -lm mixin --t_cycsz fm         # stone
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_gramdeep     -lt pos     -lf mee -lm mixin --t_cycz gram        # stone
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_nocycx       -lt pos     -lf mee -lm mixin --a_cycx 0           # stone
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_flatenc      -lt pos     -lf mee -lm mixin --enc_flat t         # tjubd5
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_plainenc     -lt pos     -lf mee -lm mixin --enc_dilation 1     # tjubd5
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_polar        -lt polar           -lm mixin                      # stone
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_sparse       -lt pos     -lf mee -lm sparse                     # stone
python train.py -n ridi -t 3 -o transposes/multiple_ridi/pos_mixin_nopre        -lt pos     -lf mee -lm mixin -ep 0 0              # stone
