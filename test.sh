CHOICES=(83)
#NAMES=("360-SOD" "360-SSOD")
NAMES=("ODI-SOD")
for CHOICE in "${CHOICES[@]}"
do
  for NAME in "${NAMES[@]}"
  do
    PYTHONPATH=$(pwd):$PYTHONPATH  python3 test.py \
                                  --output ./results/preds_ \
                                  --model_path /model/path/ \
                                  --data_path ../data/ \
                                  --batch_size 1 \
                                  --dataset $NAMES \
                                  --local_rank 7 \
                                  --epoch $CHOICE
  done
done
