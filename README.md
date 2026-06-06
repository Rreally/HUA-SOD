# HUA-SOD

## Resources

- Paper: [Hierarchical Uncertainty-Aware Salient Object Detection for 360∘ Images via Bi-Projection Collaborative Learning](https://ieeexplore.ieee.org/abstract/document/11045424)
- Checkpoint: [Google Drive](https://drive.google.com/file/d/1OC-bqwWZmdeLEKjfyqeM_XFdVHg2r2P8/view?usp=drive_link)

## Installation

The project was tested with Python 3.9, CUDA 11.7, and PyTorch 1.13.0+cu117.

1. Create and activate a conda environment:

```bash
conda create -n hivit python=3.9.17 -y
conda activate hivit
```

2. Install PyTorch with CUDA 11.7 support:

```bash
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117
```

3. Install the main dependencies:

```bash
pip install \
  numpy==1.25.2 \
  opencv-python==4.8.0.76 \
  scipy==1.11.3 \
  timm==0.5.4 \
  yacs==0.1.8 \
  pyyaml==6.0.1 \
  tensorboardX==2.6.2.2 \
  tqdm==4.66.1 \
  termcolor==2.3.0
```

4. Verify the installation:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Checkpoint

Download the checkpoint from [Google Drive](https://drive.google.com/file/d/1OC-bqwWZmdeLEKjfyqeM_XFdVHg2r2P8/view?usp=drive_link). For testing, place it under `checkpoints/` and name it according to the epoch argument, for example:

```text
checkpoints/
  ckpt_epoch_83.pth
```

## Data

```text
data/
  360-SOD/
    360-SOD-tr/
      gts/
      imgs/
    360-SOD-te/
      gts/
      imgs/
    360-SOD-tr.txt
    360-SOD-te.txt
```

## Training and Testing

```bash
python train.py  --batch_size 2 --output output --dataset 360-SOD --local_rank 1
```

```bash
bash test.sh
```

Alternatively, run `test.py` directly with the checkpoint path:

```bash
python test.py --model_path checkpoints/ --epoch 83 --dataset 360-SOD --data_path ../data/ --batch_size 1 --local_rank 0
```

## Evaluation

We use [SOD_Evaluation_Metrics](https://github.com/zyjwuyan/SOD_Evaluation_Metrics) for SOD evaluation.

## Citation

If you find this work useful in your research, please cite:

```bibtex
@ARTICLE{11045424,
  author={Zhang, Qiudan and Ji, Kaiyu and Zhang, Jie and Wang, Xu and Pan, Zhaoqing and Jiang, Jianmin},
  journal={IEEE Transactions on Multimedia},
  title={Hierarchical Uncertainty-Aware Salient Object Detection for $360 ^{\circ }$ Images via Bi-Projection Collaborative Learning},
  year={2025},
  volume={27},
  number={},
  pages={6248-6261},
  keywords={Object detection;Distortion;Feature extraction;Semantics;Accuracy;Federated learning;Convolution;Uncertainty;Three-dimensional displays;Transforms;     $360^{\circ }$      image;bi-projection;hierarchical uncertainty;salient object detection},
  doi={10.1109/TMM.2025.3581812}}
```

## Acknowledgement

We sincerely thank the authors of the [HiViT](https://github.com/zhangxiaosong18/hivit) repository. This repository is built upon it.
