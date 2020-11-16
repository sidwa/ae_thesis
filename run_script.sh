#!/bin/bash

#dataset
data=$1

#output folder
output=$2
imagenet_dir="/shared/kgcoe-research/mil/ImageNet/"
device="cuda:1"

if [ "$data" == "" ]; then
	echo "please enter a dataset name"
	exit
elif [ "$data" == "imagenet" ]; then
	if [ "$output" == "" ]; then
		echo 1
		python3 vqvae.py --data-folder "$imagenet_dir" --dataset "$data" --output-folder "$data" --device "$device" 
	else
		echo 2
		python3 vqvae.py --data-folder "$imagenet_dir" --dataset "$data" --output-folder "$output" --device "$device"
	fi 
else
	if [ "$output" == "" ]; then
		echo 3
		python3 vqvae.py --data-folder ./data/"$data" --dataset "$data" --output-folder "$data" --device "$device"
	else
		echo 4
		python3 vqvae.py --data-folder ./data/"$data" --dataset "$data" --output-folder "$output" --device "$device"
	fi 
fi