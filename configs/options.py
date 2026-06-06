# -*- coding: utf-8 -*-
from configs.config import get_config
import argparse

def parse_option():
    parser = argparse.ArgumentParser('training script', add_help=False)

    # easy config modification
    parser.add_argument('--batch_size', type=int, default=2, help="batch size for single GPU")
    parser.add_argument('--data_path', type=str, default='../data/', help='path to dataset')
    parser.add_argument("--dataset", default="360-SOD", type=str, help="dataset to train on.")
    parser.add_argument('--height', type=int, default=512, help='image height')
    parser.add_argument('--weight', type=int, default=1024, help='image height')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--use-checkpoint', action='store_true', default=True, help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config
