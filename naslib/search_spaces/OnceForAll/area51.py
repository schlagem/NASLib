# WTF

from graph import OnceForAllSearchSpace
from ofa.model_zoo import ofa_net
import os
import torch
from torchvision import datasets, transforms
import math
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics as sr
from ofa_utils import set_running_statistics

from ofa.imagenet_classification.elastic_nn.modules.dynamic_op import DynamicBatchNorm2d


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


def eval_network(net, dataloader, dev):
    correct = 0
    sr(net, dataloader)
    net.eval()
    total = len(dataloader.dataset)
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(dev), labels.to(dev)
            output = net(images)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total * 100
    return accuracy


data_path = os.path.join('~/dataset/imagenet_1k/', 'val')
data_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(data_path, ofa_transform()),
    batch_size=250,  # ~5GB on gpu memory
    shuffle=False,
    pin_memory=True
)

ss = OnceForAllSearchSpace()
ss.set_weights()
print("set statistic")
set_running_statistics(ss, data_loader)
ss.eval()
print(ss._evaluate())

net_id = "ofa_mbv3_d234_e346_k357_w1.0"
ofa_parent = ofa_net(net_id, pretrained=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ofa_network = ofa_parent.get_active_subnet(preserve_weight=True)
ofa_network.to(device)
print(eval_network(ofa_network, data_loader, device))

ss.sample_random_architecture()
print("set statistic")
set_running_statistics(ss, data_loader)
ss.eval()
print(ss._evaluate())

depths, k, e = ss.get_active_config()
ofa_parent.set_active_subnet(ks=k, e=e, d=depths)
ofa_network = ofa_parent.get_active_subnet(preserve_weight=True)
ofa_network.to(device)
print(eval_network(ofa_network, data_loader, device))


