for file in ../extended_doublenet_search/*; do
    echo "$file"
    time python compress_classifier.py --arch extended_doublenet_cifar  ../../../data.cifar10 -p=50 --lr=0.3 --epochs=200 --compress="$file" --reset-optimizer --resume-from=$1 --name="doublenet_prune"
done