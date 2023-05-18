# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License for
# CLD-SGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import numpy as np
import argparse
import torch
import os
import joblib
from tfrecord.torch.dataset import TFRecordDataset


def main(dataset, split, tfr_path, save_path):
    assert split in {'train', 'validation'}

    if dataset == 'celeba' and split in {'train', 'validation'}:
        num_shards = {'train': 120, 'validation': 40}[split]
        tfrecord_path_template = os.path.join(tfr_path, '%s/%s-r08-s-%04d-of-%04d.tfrecords')
    else:
        raise NotImplementedError

    # create lmdb
    count = 0
    for tf_ind in range(num_shards):
        # read tf_record
        tfrecord_path = tfrecord_path_template % (split, split, tf_ind, num_shards)
        index_path = None
        description = {'shape': 'int', 'data': 'byte', 'label': 'int'}
        dataset = TFRecordDataset(tfrecord_path, index_path, description)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)

        # put the data in lmdb
        for data in loader:
            im = data['data'][0].cpu().view(256, 256, 3).numpy()                        
            # joblib.dump(im, f'/scratch/rs4070/mdm/data/celeba/celeba-joblib/{split}/{count}.npy')            
            joblib.dump(im, os.path.join(save_path, f'{split}/{count}.npy'))            

            count += 1
            if count % 100 == 0:
                print(count)

    print('added %d items to the dataset.' % count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LMDB creator using TFRecords from GLOW.')
    # experimental results
    parser.add_argument('--dataset', type=str, required=True,
                        help='dataset name', choices=['imagenet-oord_32', 'imagenet-oord_32', 'celeba'])
    parser.add_argument('--tfr_path', type=str, required=True,
                        help='location of TFRecords')
    parser.add_argument('--save_path', type=str, required=True,
                        help='target location for storing lmdb files')
    args = parser.parse_args()

    main(args.dataset, 'train', args.tfr_path, args.save_path)
    main(args.dataset, 'validation', args.tfr_path, args.save_path)
