#$DEEPTRACK_PATH
# TJUIMU
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin              -lt pos     -lf mee -lm mixin                  
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_noshal       -lt pos     -lf mee -lm mixin --a_cycsz 0      
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_nodeep       -lt pos     -lf mee -lm mixin --a_cycz 0       
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_fmshal       -lt pos     -lf mee -lm mixin --t_cycsz fm     
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_gramdeep     -lt pos     -lf mee -lm mixin --t_cycz gram    
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_nocycx       -lt pos     -lf mee -lm mixin --a_cycx 0       
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_flatenc      -lt pos     -lf mee -lm mixin --enc_flat t     
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_plainenc     -lt pos     -lf mee -lm mixin --enc_dilation 1 
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_polar        -lt polar           -lm mixin                  
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_sparse       -lt pos     -lf mee -lm sparse                 
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_nopre        -lt pos     -lf mee -lm mixin -ep 0 0          
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_seq100       -lt pos     -lf mee -lm mixin --window_size 100
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_seq300       -lt pos     -lf mee -lm mixin --window_size 300
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_seq400       -lt pos     -lf mee -lm mixin --window_size 400
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_cycsz0.01    -lt pos -lf mee -lm mixin --a_cycsz 0.01  
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_cycsz0.1     -lt pos -lf mee -lm mixin --a_cycsz 0.1        
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_cycsz1       -lt pos -lf mee -lm mixin --a_cycsz 1 
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_cycsz10      -lt pos -lf mee -lm mixin --a_cycsz 10 
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_cycsz100     -lt pos -lf mee -lm mixin --a_cycsz 100  
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_cycz0.01     -lt pos -lf mee -lm mixin --a_cycz 0.01        
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_cycz0.1      -lt pos -lf mee -lm mixin --a_cycz  0.1 
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_cycz1        -lt pos -lf mee -lm mixin --a_cycz  1
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_cycz10       -lt pos -lf mee -lm mixin --a_cycz  10     
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_cycz100      -lt pos -lf mee -lm mixin --a_cycz 100         
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_cycx0.01     -lt pos -lf mee -lm mixin --a_cycx 0.01        
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_cycx0.1      -lt pos -lf mee -lm mixin --a_cycx  0.1 
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_cycx1        -lt pos -lf mee -lm mixin --a_cycx  1       
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_cycx10       -lt pos -lf mee -lm mixin --a_cycx  10         
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_cycx100      -lt pos -lf mee -lm mixin --a_cycx 100  
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_pred0.1      -lt pos -lf mee -lm mixin --a_pred 0.1         
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_pred1        -lt pos -lf mee -lm mixin --a_pred  1          
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_pred10       -lt pos -lf mee -lm mixin --a_pred  10
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_pred100      -lt pos -lf mee -lm mixin --a_pred  100        
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_pred1000     -lt pos -lf mee -lm mixin --a_pred 1000        
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_gan0.01      -lt pos -lf mee -lm mixin --a_gan 0.01         
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_gan0.1       -lt pos -lf mee -lm mixin --a_gan   0.1        
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_gan1         -lt pos -lf mee -lm mixin --a_gan   1
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_gan10        -lt pos -lf mee -lm mixin --a_gan   10         
python train.py -n tjuimu -b 32 -e 200 -c tjuimu -t 3 -o transposes/multiple_tjuimu_h1/pos_mixin_gan100       -lt pos -lf mee -lm mixin --a_gan 100          