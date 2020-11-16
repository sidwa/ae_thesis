import os
import shutil
from tqdm import tqdm

DATA_FOLDER = "/shared/kgcoe-research/mil/Flickr8k/"

# train
with open(os.path.join(DATA_FOLDER, "Flickr_8k.trainImages.txt"), "r") as train_doc:
	for line in tqdm(train_doc):
		line = line.split("\n")[0]
		shutil.copy(os.path.join(DATA_FOLDER, "Flicker8k_Dataset", line), os.path.join(DATA_FOLDER, "train"))

print("train done")

with open(os.path.join(DATA_FOLDER, "Flickr_8k.testImages.txt"), "r") as test_doc:
	for line in tqdm(test_doc):
		line = line.split("\n")[0]
		shutil.copy(os.path.join(DATA_FOLDER, "Flicker8k_Dataset", line), os.path.join(DATA_FOLDER, "test"))

print("test done")

with open(os.path.join(DATA_FOLDER, "Flickr_8k.devImages.txt"), "r") as val_doc:
	for line in tqdm(val_doc):
		line = line.split("\n")[0]
		shutil.copy(os.path.join(DATA_FOLDER, "Flicker8k_Dataset", line), os.path.join(DATA_FOLDER, "val"))

print("val done")
print("done")