import os
import pdb
import shutil
import glob

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image, make_grid

from modules import VectorQuantizedVAE, to_scalar

# codebook dims
HIDDEN_SIZE = 4
K = 512

#number of input channels in image
NUM_CHANNELS = 3

DATA_DIR = "/shared/kgcoe-research/mil/Flickr8k/"

if HIDDEN_SIZE==256:
	CKPT_DIR = "models/imagenet/best.pt" 
elif HIDDEN_SIZE==128:
	CKPT_DIR = "models/imagenet/hs_{}/best.pt".format(HIDDEN_SIZE)
else:
	CKPT_DIR = "models/imagenet/hs_{}/best.pt".format(HIDDEN_SIZE)

EMBED_DIR = os.path.expanduser(f"~/vsepp/data/f8k_precomp_vq_{HIDDEN_SIZE}/")
txt_EMBED_DIR = os.path.expanduser(f"~/vsepp/data/f8k_precomp/")
VOCAB_DIR = os.path.expanduser(f"~/vsepp/vocab/")

if not os.path.exists(EMBED_DIR):
	print("making dirs")
	os.makedirs(EMBED_DIR)

DEVICE = "cuda"
BATCH_SIZE = 256
CAP_PER_IMG = 5


model = VectorQuantizedVAE(NUM_CHANNELS, HIDDEN_SIZE, K).to("cuda")

ckpt = torch.load(CKPT_DIR)
model.load_state_dict(ckpt)


im_transform = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(128), 
												torchvision.transforms.ToTensor()])

def compute_embedding(loader, model, device=DEVICE):
	emb = None
	for img, _ in tqdm(loader):
		img = img.to(DEVICE)
		#\with torch.no_grad:
		_, _, z_q = model(img)

		z_q = z_q.detach().cpu()
		z_q = torch.reshape(z_q, [z_q.shape[0], -1])

		z_q_ = z_q
		for _ in range(1, CAP_PER_IMG):
			z_q_ = torch.cat([z_q_, z_q], axis=0)
		z_q = z_q_

		if emb is None:
			emb = z_q
		else:
			emb = torch.cat([emb, z_q], axis=0)
		
	return emb

f8k_train_dat = torchvision.datasets.ImageFolder(root=os.path.join(DATA_DIR, "train"), transform=im_transform)
f8k_test_dat = torchvision.datasets.ImageFolder(root=os.path.join(DATA_DIR, "test"), transform=im_transform)
f8k_val_dat = torchvision.datasets.ImageFolder(root=os.path.join(DATA_DIR, "val"), transform=im_transform)

f8k_train_loader = torch.utils.data.DataLoader(f8k_train_dat, batch_size=BATCH_SIZE, shuffle=False)
f8k_test_loader = torch.utils.data.DataLoader(f8k_test_dat, batch_size=BATCH_SIZE, shuffle=False)
f8k_val_loader = torch.utils.data.DataLoader(f8k_val_dat, batch_size=BATCH_SIZE, shuffle=False)

model.eval()
train_emb = compute_embedding(f8k_train_loader, model)
print(train_emb.shape)
np.save(os.path.join(EMBED_DIR, "train_ims.npy"), train_emb)

test_emb = compute_embedding(f8k_test_loader, model)
print(test_emb.shape)
np.save(os.path.join(EMBED_DIR, "test_ims.npy"), test_emb)

val_emb = compute_embedding(f8k_val_loader, model)
print(val_emb.shape)
np.save(os.path.join(EMBED_DIR, "dev_ims.npy"), val_emb)

# copy the text embeddings
txt_embed_files = glob.glob(os.path.join(txt_EMBED_DIR, "*.txt"))
for f in txt_embed_files:
	print(f)
	shutil.copy(f, EMBED_DIR)

# copy the vocab pkl
shutil.copy(os.path.join(VOCAB_DIR, "f8k_precomp_vocab.pkl"), os.path.join(VOCAB_DIR, f"f8k_precomp_vq_{HIDDEN_SIZE}_vocab.pkl"))

