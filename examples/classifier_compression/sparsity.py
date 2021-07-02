from distiller.norms import kernels_lp_norm
import torch

import os
import ipdb
import sys

import matplotlib.pyplot as plt

REGULIZED_CONV_WEIGHTS = [
    'module.layer1.0.conv1.weight',
    'module.layer1.1.conv1.weight',
    'module.layer1.2.conv1.weight',
    'module.layer2.0.conv1.weight',
    'module.layer2.1.conv1.weight',
    'module.layer2.2.conv1.weight',
    'module.layer3.0.conv1.weight',
    'module.layer3.1.conv1.weight',
    'module.layer3.2.conv1.weight',
]

if __name__ == '__main__':
    ckpt_path = sys.argv[1]
    save_file = sys.argv[2]

    torch.set_printoptions(precision=2)

    # ckpt = torch.load(os.path.join(pth, log_dir, ckpt_name))
    ckpt = torch.load(ckpt_path)

    with open(save_file, 'w+') as f:
        pass
    
    for key, value in ckpt['state_dict'].items():
        if key in REGULIZED_CONV_WEIGHTS:
            # print('l1 norm for layer {}'.format(key))
            record = '{}: {}'.format(key, kernels_lp_norm(value, group_len=value.shape[1]))
            with open(save_file, 'a') as f:
                f.write(record+'\n\n')
            # plt.figure()
            # plt.plot(kernels_lp_norm(value, group_len=value.shape[1]).cpu())
            # plt.savefig(key+'.png')
