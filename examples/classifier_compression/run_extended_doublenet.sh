# ssl filter removal training
# # extended_doublenet_cifar
python compress_classifier.py --arch extended_doublenet_cifar ../../../data.cifar10 -p=50 --lr=0.3 --epochs=250 --compress=../extended_doublenet/filter-removal_training.yaml -j=1 --deterministic --name="extended_doublenet_ssl_filter_training" --gpus 0,1
# # extended_doublenet_indep_cifar
python compress_classifier.py --arch extended_doublenet_indep_cifar ../../../data.cifar10 -p=50 --lr=0.3 --epochs=250 --compress=../extended_doublenet/filter-removal_training.yaml -j=1 --deterministic --name="extended_doublenet_indep_ssl_filter_training" --gpus 0,1
# # extended_doublenet_conn_cifar
python compress_classifier.py --arch extended_doublenet_conn_cifar ../../../data.cifar10 -p=50 --lr=0.3 --epochs=250 --compress=../extended_doublenet/filter-removal_training.yaml -j=1 --deterministic --name="extended_doublenet_ssl_conn_filter_training" --gpus 0,1

# resuming and pruning
time python compress_classifier.py --arch extended_doublenet_cifar  ../../../data.cifar10 -p=50 --lr=0.3 --epochs=200 --compress=../extended_doublenet/filter_rank.1.yaml --reset-optimizer --resume-from=logs/ --name="doublenet_prune" --gpus 0

# sensitivity analysis
time python compress_classifier.py -a doublenet_cifar ../../../data.cifar10/ -j=1 --resume=logs/doublenet_ssl_filter_training___ --sense=filter
