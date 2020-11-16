#!/bin/bash

depths="shallow moderate_shallow moderate";
autoencoder=$1

device="cuda:0"

if [[ "$autoencoder" == "vae.py" ]]; then
	batch_size=64
elif [[ "$autoencoder" == "vqvae.py" ]]; then
	batch_size=16
elif [[ "$autoencoder" == "acai.py" ]]; then
	batch_size=8
fi

echo $1 
echo $batch_size
latents=(128 64 48 32)

for enc_depth in $depths; do
	for dec_depth in $depths; do
		for latent in $latents; do
			echo "Training ${autoencoder} ||| enc type: $enc_depth ||| dec type: $dec_depth ||| batch_size: ${batch_size[$idx]}"
			echo "args --enc_type $enc_depth --dec_type $dec_depth --device $device --batch_size ${batch_size}"
			python "${autoencoder}" --enc_type $enc_depth --dec_type $dec_depth --hidden-size "$latent" --device "$device" --batch_size "${batch_size}" 2>> "$1_err.txt" || echo "error!"
		done
	done 
done
