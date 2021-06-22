from distiller.norms import kernels_lp_norm
import torch

import os
import ipdb
import sys

CONV_WEIGHTS = [
    'module.layer1.0.conv1.weight', 
    'module.layer1.0.conv2.weight', 
    'module.layer1.1.conv1.weight', 
    'module.layer1.1.conv2.weight', 
    'module.layer1.2.conv1.weight', 
    'module.layer1.2.conv2.weight', 
    'module.layer2.0.conv1.weight', 
    'module.layer2.0.conv2.weight', 
    'module.layer2.1.conv1.weight', 
    'module.layer2.1.conv2.weight', 
    'module.layer2.2.conv1.weight', 
    'module.layer2.2.conv2.weight', 
    'module.layer2.0.conv1.weight', 
    'module.layer2.0.conv2.weight', 
    'module.layer2.1.conv1.weight', 
    'module.layer2.1.conv2.weight', 
    'module.layer2.2.conv1.weight', 
    'module.layer2.2.conv2.weight', 
    'module.br1.0.weight',
    'module.br2.0.weight',
    'module.br3.0.weight',
    ]

if __name__ == '__main__':
    pth = '/home/ru4n6/Documents/distiller/examples/classifier_compression/logs'
    # log_dir = 'auxnet_training___2021.05.06-014647'
    # ckpt_name = 'auxnet_training_best.pth.tar'
    ckpt_path = sys.argv[1]

    # ckpt = torch.load(os.path.join(pth, log_dir, ckpt_name))
    ckpt = torch.load(ckpt_path)
    
    for key, value in ckpt['state_dict'].items():
        if key in CONV_WEIGHTS:
            # print('l1 norm for layer {}'.format(key))
            print('{}: {}'.format(key, kernels_lp_norm(value, group_len=value.shape[1])))
