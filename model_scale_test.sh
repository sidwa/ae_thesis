#!/bin/bash

depths="shallow moderate_shallow moderate deep";
autoencoders=(vae.py vqvae.py acai.py)
device="cuda:0"
batch_size=(64 128 32)

for enc_depth in $depths; do
	for dec_depth in $depths; do
		for idx in ${!autoencoders[@]}; do
			echo "Training ${autoencoders[$idx]} ||| enc type: $enc_depth ||| dec type: $dec_depth ||| batch_size: ${batch_size[$idx]}"
			echo "args ${autoencoders[$idx]} --enc_type $enc_depth --dec_type $dec_depth --device $device --batch_size ${batch_size[$idx]}"
			python "${autoencoders[$idx]}" --enc_type $enc_depth --dec_type $dec_depth --device "$device" --batch_size "${batch_size[$idx]}" 2>> err.txt || echo "error!"
		done
	done 
done