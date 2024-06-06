# transposemt2
python train.py -seed 0 -t 1 -dev 1 -c mt -o tmp -ep 1 -e 1
python train.py -seed 0 -t 1 -dev 3 -c tp -o tmp -ep 1 -e 1

python train.py -seed 0 -t 3 -dev 3 -c tp -o transposemt2/tp_diswei -dw t                       # tjubd12 tjubd9

python train.py -seed 0 -t 3 -dev 3 -c tp gru -o transposemt2/tp_gru                            # tjubd12
python train.py -seed 0 -t 3 -dev 1 -c tp -o transposemt2/tp_emb -el t                          # tjubd9
python train.py -seed 0 -t 3 -dev 1 -c tp -o transposemt2/tp_tsrandom --t_s_random t            # tjubd9
python train.py -seed 0 -t 3 -dev 1 -c tp -o transposemt2/tp_emb_tsrandom -el t --t_s_random t  # tjubd12

python train.py -seed 0 -t 3 -dev 1 -c tp -o transposemt2/tp_pbi -pbi t             # tjubd11
python train.py -seed 0 -t 3 -dev 2 -c tp -o transposemt2/tp_pbipl -pbi t -pl t     # tjubd11
python train.py -seed 0 -t 3 -dev 3 -c tp -o transposemt2/tp_dibi -dibi t -adv jse  # tjubd11

python train.py -seed 0 -t 3 -dev 0 -c tp -o transposemt2/tp_dibidl -dibi t -dl t -adv jse  # tjubd9
python train.py -seed 0 -t 3 -dev 1 -c tp -o transposemt2/tp_pldl -pl t -dl t -adv jse      # tjubd9

python train.py -seed 0 -t 3 -dev 2 -c tp -o transposemt2/tp_dibidl_drop2 -dibi t -dl t -adv jse -d 0.2 # tjubd12

python train.py -seed 0 -t 3 -dev 1 -c tp -o transposemt2/tp_pbidibi -pbi t -dibi t -adv jse                 # tjubd12
python train.py -seed 0 -t 3 -dev 3 -c tp -o transposemt2/tp_pbipldibidl -pbi t -pl t -dibi t -dl t -adv jse # tjubd12

python gather_result2.py -n tjubd11 -o tp_pbi tp_pbipl tp_dibi
python gather_result2.py -n tjubd9 -o tp_dibidl tp_pldl tp_drop2 tp_drop1 tp_emb tp_tsrandom
python gather_result2.py -n tjubd12 -o tp_pbidibi tp_pbipldibidl tp_gru tp_dibidl_drop2 tp_emb_tsrandom

python train.py -seed 0 -t 3 -dev 1 -c mt -o transposemt2/mt_base                   # tjubd12
python train.py -seed 0 -t 3 -dev 3 -c tp -o transposemt2/tp_base                   # tjubd12

python train.py -seed 0       -t 5  -dev 1 -c mt -o transposemt2/mt_base                   # tjubd5
python train.py -seed 10 -st 5 -t 10 -dev 0 -c mt -o transposemt2/mt_base                   # tjubd5

python train.py -seed 0 -t 3 -dev 0 -c tp -o transposemt2/tp_act -act t             # stone
python train.py -seed 0 -t 3 -dev 0 -c tp -o transposemt2/tp_drop1 -d 0.1           # tjubd9
python train.py -seed 0 -t 3 -dev 1 -c tp -o transposemt2/tp_drop2 -d 0.2           # tjubd9

python train.py -seed 0 -t 3 -dev 3 -c tp -o transposemt2/tp_ebi -ebi t             # tjubd11
python train.py -seed 0 -t 3 -dev 0 -c tp -o transposemt2/tp_debigbi -debi t -gbi t # tjubd11
python train.py -seed 0 -t 3 -dev 1 -c tp -o transposemt2/tp_pbi -pbi t             # tjubd11
python train.py -seed 0 -t 3 -dev 2 -c tp -o transposemt2/tp_dibi -dibi t           # tjubd11
python train.py -seed 0 -t 3 -dev 1 -c tp -o transposemt2/tp_pbidibi -pbi t -dibi t # tjubd9

python train.py -seed 0 -t 3 -dev 0 -c tp -o transposemt2/tp_genflat -gf t          # tjubd5
python train.py -seed 0 -t 3 -dev 1 -c tp -o transposemt2/tp_predflat -pf t         # tjubd5
python train.py -seed 0 -t 3 -dev 0 -c tp -o transposemt2/tp_decnoflat -def f       # stone

python train.py -seed 0 -t 3 -dev 3 -c tp -o transposemt2/tp_wgan -adv w            # tjubd11
python train.py -seed 0 -t 3 -dev 2 -c tp -o transposemt2/tp_lsgan -adv ls          # tjubd11


python gather_result2.py -n tjubd11 -o tp_ebi tp_debigbi tp_pbi tp_pbidibi tp_wgan tp_lsgan
python gather_result2.py -n tjubd9 -o tp_pbidibi
python gather_result2.py -n tjubd5 -o tp_drop1  tp_drop2  tp_genflat  tp_predflat
python gather_result2.py -n stone -o tp_act  tp_decnoflat

# transposemt1
python train.py -seed 0 -t 3 -dev 0 -c mt -o transposemt1/mtbase # tjubd9
python train.py -seed 0 -t 3 -dev 3 -c mt -o transposemt1/mt_nocond -ce f # tjubd12
python train.py -seed 0 -t 3 -dev 2 -c mt -o transposemt1/mt_bi -bi t # tjubd11
python train.py -seed 0 -t 3 -dev 3 -c tp -o transposemt1/tp_nobidi -bi f # tjubd12
python train.py -seed 0 -t 3 -dev 3 -c tp -o transposemt1/tp_lstmenc -et lstm # tjubd11
python train.py -seed 0 -t 3 -dev 1 -c tp -o transposemt1/tp_lstmenc_nobidi -et lstm -bi f # tjubd11
python train.py -seed 0 -t 3 -dev 1 -c tp -o transposemt1/tpbase # tjubd9
 
# transposemt
python train_multitarget.py -seed 0 -t 3 -dev 0 -c mt -o transposemt/mtbase
python train_multitarget.py -seed 0 -t 3 -dev 1 -c mt -o transposemt/mt_nocond -ce f
python train_multitarget.py -seed 0 -t 3 -dev 0 -c tp -o transposemt/tp_cycsz0 --a_cycsz 0
python train_multitarget.py -seed 0 -t 3 -dev 1 -c tp -o transposemt/tpbase

python train_multitarget.py -seed 0 -t 3 -dev 1 -c tp -o transposemt/tp_efulstm         -efu lstm -z 8 -ep 1
python train_multitarget.py -seed 0 -t 3 -dev 0 -c tp -o transposemt/tp_efulstmExp1     -efu lstm -z 8 -ep 1 -eex 1

python train_multitarget.py -seed 0 -t 3 -dev 0 -c mt -o transposemt/mt_predind -pf f

python gather_result2.py -n tjubd9 -o mtbase mt_nocond tpbase tp_cycsz0
python gather_result2.py -n stone -o tp_dense
python gather_result2.py -n tjubd12 -o tp_dense_noshal tp_dense_nodeep tp_dense_fmshal tp_dense_grdeep
python gather_result2.py -n tjubd5 -o tp_norm16 tp_norm64

python train_multitarget.py -seed 0 -t 3 -dev 0 -c tp -o transposemt/tp_norm16 -pn t -pz 16 # stone
python train_multitarget.py -seed 0 -t 3 -dev 1 -c tp -o transposemt/tp_norm64 -pn t -pz 64 # stone
python train_multitarget.py -seed 0 -t 3 -dev 0 -c tp -o transposemt/tp_dense -lm mix # stone

python train_multitarget.py -seed 0 -t 1 -dev 3 -c tp -o transposemt/tp_dense_noshal    -lm mix --a_cycsz 0         # tjubd12
python train_multitarget.py -seed 0 -t 1 -dev 3 -c tp -o transposemt/tp_dense_nodeep    -lm mix --a_cycz 0          # tjubd12
python train_multitarget.py -seed 0 -t 1 -dev 3 -c tp -o transposemt/tp_dense_fmshal    -lm mix --t_cycsz fm        # tjubd12
python train_multitarget.py -seed 0 -t 1 -dev 3 -c tp -o transposemt/tp_dense_grdeep    -lm mix --t_cycz gram       # tjubd12
python train_multitarget.py -seed 0 -t 1 -dev 3 -c tp -o transposemt/tp_dense_flatenc   -lm mix --enc_flat t        # tjubd12
python train_multitarget.py -seed 0 -t 1 -dev 3 -c tp -o transposemt/tp_dense_plainenc  -lm mix --enc_dilation 1    # tjubd12


# 23.01.07 label type
python train_mt.py -e 50 -t 3 -du 1.0 -lt polar -lf mse -lm sparse -tp 1 -o mt/polar_sparse1 -seed 0
python train_mt.py -e 50 -t 3 -du 1.0 -lt polar -lf mse -lm sparse -tp 2 -o mt/polar_sparse2 -seed 0
python train_mt.py -e 50 -t 3 -du 1.0 -lt polar -lf mse -lm sparse -tp 3 -o mt/polar_sparse3 -seed 0

python train.py     -e 50 -t 3  --data_usage 1.0 -lt polar  -lf mse -lm sparse              -seed 0 -o transpose/polar_sparse               -dev 2
python train.py     -e 50 -t 3  --data_usage 1.0 -lt polar  -lf mse -lm dense               -seed 0 -o transpose/polar_dense                -dev 2
python train.py     -e 50 -t 3  --data_usage 1.0 -lt polar  -lf mse -lm dense        -ens t -seed 0 -o transpose/polar_dense_ensemble       -dev 2
python train.py     -e 50 -t 3  --data_usage 1.0 -lt offset -lf mee -lm sparse              -seed 0 -o transpose/offset_sparse              -dev 2
python train.py     -e 50 -t 3  --data_usage 1.0 -lt offset -lf mee -lm sparse -pr t        -seed 0 -o transpose/offset_sparse_proj         -dev 2
python train.py     -e 50 -t 3  --data_usage 1.0 -lt offset -lf mee -lm dense               -seed 0 -o transpose/offset_dense               -dev 2
python train.py     -e 50 -t 3  --data_usage 1.0 -lt offset -lf mee -lm dense  -pr t        -seed 0 -o transpose/offset_dense_proj          -dev 2
python train.py     -e 50 -t 3  --data_usage 1.0 -lt offset -lf mee -lm dense  -pr t -ens t -seed 0 -o transpose/offset_dense_proj_ensemble -dev 2
python train.py     -e 50 -t 3  --data_usage 1.0 -lt offset -lf mee -lm dense  -pr f -ens t -seed 0 -o transpose/offset_dense_ensemble      -dev 2

# abla
python train.py -t 3 -o transposes/multiple/pos_mixin_2          -lt pos     -lf mee -lm mixin                      # tjubd5
python train.py -t 3 -o transposes/multiple/pos_mixin_noshal     -lt pos     -lf mee -lm mixin --a_cycsz 0          # tjubd5
python train.py -t 3 -o transposes/multiple/pos_mixin_nodeep     -lt pos     -lf mee -lm mixin --a_cycz 0           # stone
python train.py -t 3 -o transposes/multiple/pos_mixin_fmshal     -lt pos     -lf mee -lm mixin --t_cycsz fm         # stone
python train.py -t 3 -o transposes/multiple/pos_mixin_gramdeep   -lt pos     -lf mee -lm mixin --t_cycz gram        # stone
python train.py -t 3 -o transposes/multiple/pos_mixin_nocycx     -lt pos     -lf mee -lm mixin --a_cycx 0           # stone
python train.py -t 3 -o transposes/multiple/pos_mixin_flatenc    -lt pos     -lf mee -lm mixin --enc_flat t         # tjubd5
python train.py -t 3 -o transposes/multiple/pos_mixin_plainenc   -lt pos     -lf mee -lm mixin --enc_dilation 1     # tjubd5
python train.py -t 3 -o transposes/multiple/pos_mixin_polar      -lt polar           -lm mixin                      # stone
python train.py -t 3 -o transposes/multiple/pos_mixin_sparse     -lt pos     -lf mee -lm sparse                     # stone
python train.py -t 3 -o transposes/multiple/pos_mixin_nopre      -lt pos     -lf mee -lm mixin -ep 0 0              # stone
python train.py -t 3 -o transposes/multiple/pos_mixin_seq100     -lt pos     -lf mee -lm mixin --window_size 100    # tjubd5
python train.py -t 3 -o transposes/multiple/pos_mixin_seq300     -lt pos     -lf mee -lm mixin --window_size 300    # tjubd5
python train.py -t 3 -o transposes/multiple/pos_mixin_seq400     -lt pos     -lf mee -lm mixin --window_size 400    # tjubd5
python train.py -t 3 -o transposes/multiple/pos_mixin_cycsz0.1  -lt pos -lf mee -lm mixin --a_cycsz 0.1             # tjubd5
python train.py -t 3 -o transposes/multiple/pos_mixin_cycz0.1   -lt pos -lf mee -lm mixin --a_cycz  0.1             # tjubd5
python train.py -t 3 -o transposes/multiple/pos_mixin_cycx0.1   -lt pos -lf mee -lm mixin --a_cycx  0.1             # tjubd5
python train.py -t 3 -o transposes/multiple/pos_mixin_pred1     -lt pos -lf mee -lm mixin --a_pred  1               # tjubd5
python train.py -t 3 -o transposes/multiple/pos_mixin_gan0.1    -lt pos -lf mee -lm mixin --a_gan   0.1             # tjubd5
python train.py -t 3 -o transposes/multiple/pos_mixin_cycsz10   -lt pos -lf mee -lm mixin --a_cycsz 10              # stone
python train.py -t 3 -o transposes/multiple/pos_mixin_cycz10    -lt pos -lf mee -lm mixin --a_cycz  10              # stone
python train.py -t 3 -o transposes/multiple/pos_mixin_cycx10    -lt pos -lf mee -lm mixin --a_cycx  10              # stone
python train.py -t 3 -o transposes/multiple/pos_mixin_pred100   -lt pos -lf mee -lm mixin --a_pred  100             # stone
python train.py -t 3 -o transposes/multiple/pos_mixin_gan10     -lt pos -lf mee -lm mixin --a_gan   10              # stone
python train.py -t 3 -o transposes/multiple/pos_mixin_cycsz0.01 -lt pos -lf mee -lm mixin --a_cycsz 0.01            # stone
python train.py -t 3 -o transposes/multiple/pos_mixin_cycsz100  -lt pos -lf mee -lm mixin --a_cycsz 100             # stone
python train.py -t 3 -o transposes/multiple/pos_mixin_cycz0.01  -lt pos -lf mee -lm mixin --a_cycz 0.01             # stone
python train.py -t 3 -o transposes/multiple/pos_mixin_cycz100   -lt pos -lf mee -lm mixin --a_cycz 100              # stone
python train.py -t 3 -o transposes/multiple/pos_mixin_cycx0.01  -lt pos -lf mee -lm mixin --a_cycx 0.01             # stone
python train.py -t 3 -o transposes/multiple/pos_mixin_cycx100   -lt pos -lf mee -lm mixin --a_cycx 100              # stone
python train.py -t 3 -o transposes/multiple/pos_mixin_pred0.1  -lt pos -lf mee -lm mixin --a_pred 0.1               # tjubd
python train.py -t 3 -o transposes/multiple/pos_mixin_pred1000 -lt pos -lf mee -lm mixin --a_pred 1000              # tjubd
python train.py -t 3 -o transposes/multiple/pos_mixin_gan0.01  -lt pos -lf mee -lm mixin --a_gan 0.01               # tjubd
python train.py -t 3 -o transposes/multiple/pos_mixin_gan100   -lt pos -lf mee -lm mixin --a_gan 100                # tjubd

# test label
python train_transpose_mt.py    -o label       -tp 2
# test model
python train_mt.py -o mt_tar1       -tp 1
python train_mt.py -o mt_tar2       -tp 2
python train_mt.py -o mt_tar3       -tp 3

python train_mt.py -o mt_tar1_fn    -tp 1 -fn True
python train_mt.py -o mt_tar2_fn    -tp 2 -fn True
python train_mt.py -o mt_tar3_fn    -tp 3 -fn True

python train_transpose_mt.py    -o transpose_mt_tar1       -tp 1
python train_transpose_mt.py    -o transpose_mt_tar2       -tp 2
python train_transpose_mt.py    -o transpose_mt_tar3       -tp 3

python train_transpose_mt.py    -o transpose_mt_tar1_pos_mix       -tp 1 -lt pos -lf mee -lm mix -ep 2
python train_transpose_mt.py    -o transpose_mt_tar2_pos_mix       -tp 2 -lt pos -lf mee -lm mix -ep 2
python train_transpose_mt.py    -o transpose_mt_tar3_pos_mix       -tp 3 -lt pos -lf mee -lm mix -ep 2

python train_transpose_mt.py    -o transpose_mt_tar1_pos_sparse       -tp 1 -lt pos -lf mee -ep 2
python train_transpose_mt.py    -o transpose_mt_tar2_pos_sparse       -tp 2 -lt pos -lf mee -ep 2
python train_transpose_mt.py    -o transpose_mt_tar3_pos_sparse       -tp 3 -lt pos -lf mee -ep 2

python train_transpose_mt.py    -o transpose_mt_tar1_inpolar_mix       -tp 1 -lt inpolar -lm mix -ep 2
python train_transpose_mt.py    -o transpose_mt_tar2_inpolar_mix       -tp 2 -lt inpolar -lm mix -ep 2
python train_transpose_mt.py    -o transpose_mt_tar3_inpolar_mix       -tp 3 -lt inpolar -lm mix -ep 2

python train_transpose_mt.py    -o transposes/single/T1_pos_sparse_1    -tp 1 -lt pos -lf mee -lm sparse -ep 2 -dev 0 --data_usage 1
python train_transpose_mt.py    -o transposes/single/T2_pos_sparse_1    -tp 2 -lt pos -lf mee -lm sparse -ep 2 -dev 0 --data_usage 1

python train_transpose_mt.py    -o transposes/single/T1_pos_mix_1       -tp 1 -lt pos -lf mee -lm mix -ep 2 -dev 0 --data_usage 1
python train_transpose_mt.py    -o transposes/single/T2_pos_mix_1       -tp 2 -lt pos -lf mee -lm mix -ep 2 -dev 0 --data_usage 1
python train_transpose_mt.py    -o transposes/single/T3_pos_mix_1       -tp 3 -lt pos -lf mee -lm mix -ep 2 -dev 0 --data_usage 1

python train_transposeS.py      -o label/polar_sparse   # -load True -tn -1
python train_transposeS.py      -o label/inpolar_sparse -lt inpolar
python train_transposeS.py      -o label/pos_sparse     -lt pos -lf mee

python train_transposeS.py      -o label/polar_dense    -lm dense
python train_transposeS.py      -o label/inpolar_dense  -lt inpolar -lm dense
python train_transposeS.py      -o label/pos_dense      -lt pos -lf mee -lm dense

python train_transposeS.py      -o label/polar_mix    -lm mix
python train_transposeS.py      -o label/inpolar_mix  -lt inpolar -lm mix
python train_transposeS.py      -o label/pos_mix      -lt pos -lf mee -lm mix

python train.py                 -o transpose_mt2            -ep 2
python train.py                 -o transposes/pos_mix       -lt pos -lf mee -lm mix
python train.py                 -o transposes/polar_sparse  -lt polar       -lm sparse
