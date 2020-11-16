#!/bin/bash

#dataset
data="imagenet"

#output folder
output=$2
imagenet_dir="/shared/kgcoe-research/mil/ImageNet/"
device="cuda:1"

for hs in 128 64 48 32
do
	#echo $hs
	python3 vqvae.py --data-folder "/shared/kgcoe-research/mil/ImageNet/" --dataset "imagenet" --hidden-size $hs --output-folder "imagenet/hs_$hs/" --device "cuda:0"
done

python3 vqvae.py --data-folder "/shared/kgcoe-research/mil/ImageNet/" --dataset "imagenet" --deep_net --hidden-size 256 --output-folder "imagenet/deep_hs_256/" --device "cuda:0" 
python3 vqvae.py --data-folder "/shared/kgcoe-research/mil/ImageNet/" --dataset "imagenet" --deep_net --hidden-size 128 --output-folder "imagenet/deep_hs_128/" --device "cuda:0" 
python3 vqvae.py --data-folder "/shared/kgcoe-research/mil/ImageNet/" --dataset "imagenet" --deep_net --hidden-size 64 --output-folder "imagenet/deep_hs_64/" --device "cuda:0" 
python3 vqvae.py --data-folder "/shared/kgcoe-research/mil/ImageNet/" --dataset "imagenet" --deep_net --hidden-size 48 --output-folder "imagenet/deep_hs_48/" --device "cuda:0" 
python3 vqvae.py --data-folder "/shared/kgcoe-research/mil/ImageNet/" --dataset "imagenet" --deep_net --hidden-size 32 --output-folder "imagenet/deep_hs_32/" --device "cuda:0"

python3 vqvae.py --data-folder "/shared/kgcoe-research/mil/ImageNet/" --dataset "imagenet" --img_res 32 --hidden-size 256 --output-folder "imagenet/hs_32_4/" --device "cuda:0" 
