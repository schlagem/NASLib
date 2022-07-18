# WTF

from graph import OnceForAllSearchSpace
from ofa.model_zoo import ofa_net
import os
import torch
from torchvision import datasets, transforms
import math

ss = OnceForAllSearchSpace()
ss.set_weights()
# ss.eval()

print(ss._evaluate())

net_id = "ofa_mbv3_d234_e346_k357_w1.0"
ofa_network = ofa_net(net_id, pretrained=True)
ofa_network.cuda()
# ofa_network.eval()


def ofa_transform(image_size=None):
    if image_size is None:
        image_size = 224
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    return transforms.Compose([
        transforms.Resize(int(math.ceil(image_size / 0.875))),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize]
    )


data_path = os.path.join('~/dataset/imagenet_1k/', 'val')
data_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(data_path, ofa_transform()),
    batch_size=250,  # ~5GB on gpu memory
    shuffle=False,
    num_workers=16,
    pin_memory=True
)
correct = 0
total = len(data_loader.dataset)
# self.to(self.device)
with torch.no_grad():
    for images, labels in data_loader:
        images, labels = images.cuda(), labels.cuda()
        output = ofa_network(images)
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == labels).sum().item()
accuracy = correct / total * 100
print(accuracy)

ss.sample_random_architecture()
print(ss._evaluate())

depths, k, e = ss.get_active_config()
ofa_network.set_active_subnet(ks=k, e=e, d=depths)
ofa_network = ofa_network.get_active_subnet()
ofa_network.eval()
correct = 0
total = len(data_loader.dataset)
# self.to(self.device)
with torch.no_grad():
    for images, labels in data_loader:
        images, labels = images.cuda(), labels.cuda()
        output = ofa_network(images)
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == labels).sum().item()
accuracy = correct / total * 100
print(accuracy)
