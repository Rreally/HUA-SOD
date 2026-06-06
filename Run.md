<!-- 将Uncertain部分的特征提取器替换为HiVit -->

1. env:
    source activate hivit

2. train:
    python train.py  --batch_size 2 --output output --dataset 360-SOD --local_rank 1

3. test:
    1) python test.py  --output ./results/pred_output_wffc/preds_ --model_path /data/A_GGDANet/output/ --batch_size 1 --epoch 75 
    2) bash test.sh

4. eval:
    cd /data/sod_eval
    ln -s /data/A_GGDANet/pred_maps ./
    python main.py