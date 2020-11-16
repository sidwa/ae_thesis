import os

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from modules import VAE

data_folder = "/shared/kgcoe-research/mil/ImageNet/"
img_res = 128
batch_size = 10000

im_transform = transforms.Compose([ \
					transforms.CenterCrop(img_res), \
					transforms.ToTensor()])
train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_folder, "train"), transform=im_transform )
test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_folder, "val"), transform=im_transform )
valid_dataset = test_dataset

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

train_loader = torch.utils.data.DataLoader(train_dataset,
		batch_size=batch_size, shuffle=True,
		num_workers=0, pin_memory=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset,
		batch_size=batch_size, shuffle=False, drop_last=True,
		num_workers=0, pin_memory=True)

model = VAE(3, 256).to("cuda")
opt = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)


model.train()
for batch_idx, (x, _) in enumerate(train_loader):
	x = x.to("cuda")
	x_tilde, kl_d = model(x)
	loss_recons = F.mse_loss(x_tilde, x, reduction="sum") / x.size(0)
	loss = loss_recons + kl_d

		# nll = -Normal(x_tilde, torch.ones_like(x_tilde)).log_prob(x)
		# log_px = nll.mean().item() - np.log(128) + kl_d.item()
		# log_px /= np.log(2)

	opt.zero_grad()
	loss.backward()
	opt.step()

