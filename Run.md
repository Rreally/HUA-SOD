<!-- 将Uncertain部分的特征提取器替换为HiVit -->

1. env:
    source activate hivit

2. train:
    python train.py  --batch_size 2 --output output --dataset 360-SOD --local_rank 1

3. test:
    bash test.sh