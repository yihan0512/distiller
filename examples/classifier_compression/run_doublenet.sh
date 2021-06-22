# resnet56 baseline training
# time python compress_classifier.py --arch resnet56_cifar  ../../../data.cifar10 -p=50 --lr=0.3 --epochs=180 --compress=../pruning_filters_for_efficient_convnets/resnet56_cifar_baseline_training.yaml -j=1 --deterministic

# kd training with teacher resnet56 and resnet20 student
# time python compress_classifier.py --kd-teacher resnet56_cifar --kd-resume logs/2021.03.25-141952/best.pth.tar --kd-start-epoch 0 -a resnet20_cifar ../../../data.cifar10/

################################################## doublenet #########################################

# doublenet baseline training w/o lr scheduler
# time python compress_classifier.py --arch doublenet_cifar ../../../cifar-data/cifar-10-batches-py/ -p 30 -j=1 --lr=0.01 --epochs 800 --wd 0.001

# doublenet baseline training w/ lr scheduler
# time python compress_classifier.py --arch doublenet_cifar ../../../data.cifar10 -p=50 --lr=0.3 --epochs=300 --compress=../doublenet/doublenet_cifar_baseline_training.yaml -j=1 --deterministic --name="doublenet_training" --gpus 1

# doublenet resuming and finetuning
# time python compress_classifier.py --arch doublenet_cifar ../../../data.cifar10 -p=50 --lr=0.3 --epochs=70 --compress=../doublenet/doublenet_cifar_baseline_training.yaml -j=1 --deterministic --name="doublenet_resume" --resume=logs/doublenet-best/best.pth.tar

# doublenet ssl channel removal training
# time python compress_classifier.py --arch doublenet_cifar ../../../data.cifar10 -p=50 --lr=0.3 --epochs=800 --compress=../doublenet/doublenet_cifar_channels-removal_training.yaml -j=1 --deterministic 

# doublenet ssl filter removal training 
# # doublenet_cifar
# python compress_classifier.py --arch doublenet_cifar ../../../data.cifar10 -p=50 --lr=0.3 --epochs=250 --compress=../doublenet/doublenet_cifar_filter-removal_training.1.yaml -j=1 --deterministic --name="doublenet_ssl_filter_training"
# # doublenet_cifar_shortcuts_pretrained
# python compress_classifier.py --arch doublenet_cifar_shortcuts_pretrained ../../../data.cifar10 -p=50 --lr=0.3 --epochs=250 --compress=../doublenet/doublenet_cifar_filter-removal_training.yaml -j=1 --deterministic --name="doublenet_shortcuts_pretrained_ssl_filter_training" --gpus 0
# # doublenet_cifar_deepnet_pretrained
# python compress_classifier.py --arch doublenet_cifar_deepnet_pretrained ../../../data.cifar10 -p=50 --lr=0.3 --epochs=250 --compress=../doublenet/doublenet_cifar_filter-removal_training.yaml -j=1 --deterministic --name="doublenet_deepnet_pretrained_ssl_filter_training" --gpus 0
# # doublenet_indep_cifar
# python compress_classifier.py --arch doublenet_indep_cifar ../../../data.cifar10 -p=50 --lr=0.3 --epochs=250 --compress=../doublenet/doublenet_indep_cifar_filter-removal_training.yaml -j=1 --deterministic --name="doublenet_indep_ssl_filter_training"
# # doublenet_conn_cifar
# python compress_classifier.py --arch doublenet_conn_cifar ../../../data.cifar10 -p=50 --lr=0.3 --epochs=250 --compress=../doublenet/doublenet_conn_cifar_filter-removal_training.yaml -j=1 --deterministic --name="doublenet_conn_ssl_filter_training"


# doublenet ssl filter removal training and pruning
# time python compress_classifier.py --arch doublenet_cifar ../../../data.cifar10 -p=50 --lr=0.3 --epochs=300 --compress=../doublenet/doublenet_cifar_filter-removal_training_pruning.yaml -j=1 --deterministic --name="ssl_filter_prune"

# doublenet resumeing full model pruning
# time python compress_classifier.py --arch doublenet_cifar ../../../data.cifar10 -p=50 --lr=0.03 --epochs=70 --compress=../doublenet/doublenet_cifar_filter_rank.yaml -j=1 --deterministic --name="doublenet_full_prune" --resume=logs/doublenet-best/best.pth.tar
# time python compress_classifier.py --arch doublenet_cifar_shortcuts_pretrained ../../../data.cifar10 -p=50 --lr=0.03 --epochs=70 --compress=../doublenet/doublenet_cifar_filter_rank.yaml -j=1 --deterministic --name="doublenet_full_prune" --resume=logs/doublenet_shortcuts_pretrained-best/best.pth.tar --gpus 0

# doublenet resuming pruning
# doublenet_cifar
# time python compress_classifier.py --arch doublenet_cifar  ../../../data.cifar10 -p=50 --lr=0.3 --epochs=200 --compress=../doublenet/doublenet_cifar_filter_rank.1.yaml --reset-optimizer --resume-from=logs/doublenet_ssl_filter_training___2021.05.14-022816/doublenet_ssl_filter_training_best.pth.tar --name="doublenet_prune" --gpus 0
# time python compress_classifier.py --arch doublenet_cifar  ../../../data.cifar10 -p=50 --lr=0.3 --epochs=200 --compress=../doublenet/doublenet_cifar_filter_rank.1.yaml --reset-optimizer --resume-from=logs/doublenet_ssl_filter_training___2021.05.15-031702/doublenet_ssl_filter_training_best.pth.tar --name="doublenet_prune" --gpus 0
# # doublenet_indep_cifar
# time python compress_classifier.py --arch doublenet_indep_cifar  ../../../data.cifar10 -p=50 --lr=0.3 --epochs=200 --compress=../doublenet/doublenet_indep_cifar_filter_rank.yaml --reset-optimizer --resume-from=logs/doublenet_indep_ssl_filter_training___2021.04.25-200405/doublenet_indep_ssl_filter_training_best.pth.tar --name="doublenet_indep_prune"
# # doublenet_conn_cifar
# time python compress_classifier.py --arch doublenet_conn_cifar  ../../../data.cifar10 -p=50 --lr=0.3 --epochs=200 --compress=../doublenet/doublenet_conn_cifar_filter_rank.yaml --reset-optimizer --resume-from=logs/doublenet_conn_ssl_filter_training___2021.04.25-214041/doublenet_conn_ssl_filter_training_best.pth.tar --name="doublenet_conn_prune"

# doublenet resuming pruning various sparsity
# # sparsity 0.1
# time python compress_classifier.py --arch doublenet_cifar  ../../../data.cifar10 -p=50 --lr=0.3 --epochs=200 --compress=../doublenet1/doublenet_cifar_filter_rank_1.yaml --reset-optimizer --resume-from=logs/doublenet_ssl_filter_training___2021.04.25-182706/doublenet_ssl_filter_training_best.pth.tar --name="doublenet_prune_1" --gpus 1
# # #sparsity 0.3
# time python compress_classifier.py --arch doublenet_cifar  ../../../data.cifar10 -p=50 --lr=0.3 --epochs=200 --compress=../doublenet1/doublenet_cifar_filter_rank_3.yaml --reset-optimizer --resume-from=logs/doublenet_ssl_filter_training___2021.04.25-182706/doublenet_ssl_filter_training_best.pth.tar --name="doublenet_prune_3" --gpus 1
# # #sparsity 0.5
# time python compress_classifier.py --arch doublenet_cifar  ../../../data.cifar10 -p=50 --lr=0.3 --epochs=200 --compress=../doublenet1/doublenet_cifar_filter_rank_5.yaml --reset-optimizer --resume-from=logs/doublenet_ssl_filter_training___2021.04.25-182706/doublenet_ssl_filter_training_best.pth.tar --name="doublenet_prune_5" --gpus 1
# #sparsity 0.7
# time python compress_classifier.py --arch doublenet_cifar  ../../../data.cifar10 -p=50 --lr=0.3 --epochs=200 --compress=../doublenet1/doublenet_cifar_filter_rank_7.yaml --reset-optimizer --resume-from=logs/doublenet_ssl_filter_training___2021.04.25-182706/doublenet_ssl_filter_training_best.pth.tar --name="doublenet_prune_7" --gpus 1
# #sparsity 0.9
# time python compress_classifier.py --arch doublenet_cifar  ../../../data.cifar10 -p=50 --lr=0.3 --epochs=200 --compress=../doublenet1/doublenet_cifar_filter_rank_9.yaml --reset-optimizer --resume-from=logs/doublenet_ssl_filter_training___2021.04.25-182706/doublenet_ssl_filter_training_best.pth.tar --name="doublenet_prune_9" --gpus 1



################################################## auxnet #########################################

# auxnet baseline training w/ lr scheduler
# time python compress_classifier.py --arch doublenet_cifar_shortcuts_pretrained ../../../data.cifar10 -p=50 --lr=0.3 --epochs=180 --compress=../doublenet/auxnet_cifar_baseline_training.yaml -j=1 --deterministic --name="auxnet_training"

################################################## resnet20 #########################################

# resnet20 ssl filter removal training and finetuning
# python compress_classifier.py --arch resnet20_cifar ../../../data.cifar10 -p=50 --lr=0.3 --epochs=250 --compress=../doublenet/resnet20_cifar_filter-removal_training.yaml -j=1 --deterministic --name="resnet20_ssl_filter_training"

# resnet20 ssl channel removal training
# time python compress_classifier.py --arch resnet20_cifar  ../../../data.cifar10 -p=50 --lr=0.3 --epochs=180 --compress=../ssl/ssl_channels-removal_training_x1.8.yaml -j=1 --deterministic
# time python compress_classifier.py --arch resnet20_cifar  ../../../data.cifar10 -p=50 --lr=0.2 --epochs=98 --compress=../ssl/ssl_channels-removal_finetuning.yaml -j=1 --deterministic --resume=logs/2021.04.11-190007/best.pth.tar


# resnet20 ssl filter removal training and finetuning
# time python compress_classifier.py --arch resnet20_cifar  ../../../data.cifar10 -p=50 --lr=0.3 --epochs=180 --compress=../ssl/ssl_filter-removal_training.yaml -j=1 --deterministic --name="filters"
# time python compress_classifier.py --arch resnet20_cifar  ../../../data.cifar10 -p=50 --lr=0.2 --epochs=98 --compress=../ssl/ssl_channels-removal_finetuning.yaml --reset-optimizer --resume-from=logs/filters___2021.04.11-190608/filters_best.pth.tar

# resnet20 one shot structured pruning
# time python compress_classifier.py -a=resnet20_cifar -p=50 ../../../data.cifar10 --epochs=70 --lr=0.1 --compress=../doublenet/resnet20_cifar_filter_rank.yaml --resume-from=logs/resnet20-best-1/best.pth.tar  --reset-optimizer --vs=0

#resnet20 resuming pruning various sparsity
# # 0.1
# time python compress_classifier.py --arch resnet20_cifar  ../../../data.cifar10 -p=50 --lr=0.3 --epochs=200 --compress=../resnet20_prune/resnet20_cifar_filter_rank_1.yaml --reset-optimizer --resume-from=logs/resnet20_ssl_filter_training___2021.04.15-223901/resnet20_ssl_filter_training_best.pth.tar --name="resnet20_prune_1"
# # 0.3
# time python compress_classifier.py --arch resnet20_cifar  ../../../data.cifar10 -p=50 --lr=0.3 --epochs=200 --compress=../resnet20_prune/resnet20_cifar_filter_rank_3.yaml --reset-optimizer --resume-from=logs/resnet20_ssl_filter_training___2021.04.15-223901/resnet20_ssl_filter_training_best.pth.tar --name="resnet20_prune_3"
# # 0.5
# time python compress_classifier.py --arch resnet20_cifar  ../../../data.cifar10 -p=50 --lr=0.3 --epochs=200 --compress=../resnet20_prune/resnet20_cifar_filter_rank_5.yaml --reset-optimizer --resume-from=logs/resnet20_ssl_filter_training___2021.04.15-223901/resnet20_ssl_filter_training_best.pth.tar --name="resnet20_prune_5"
# # 0.7
# time python compress_classifier.py --arch resnet20_cifar  ../../../data.cifar10 -p=50 --lr=0.3 --epochs=200 --compress=../resnet20_prune/resnet20_cifar_filter_rank_7.yaml --reset-optimizer --resume-from=logs/resnet20_ssl_filter_training___2021.04.15-223901/resnet20_ssl_filter_training_best.pth.tar --name="resnet20_prune_7"
# 0.9
# time python compress_classifier.py --arch resnet20_cifar  ../../../data.cifar10 -p=50 --lr=0.3 --epochs=200 --compress=../resnet20_prune/resnet20_cifar_filter_rank_9.yaml --reset-optimizer --resume-from=logs/resnet20_ssl_filter_training___2021.04.15-223901/resnet20_ssl_filter_training_best.pth.tar --name="resnet20_prune_9"