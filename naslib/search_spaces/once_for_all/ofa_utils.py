import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import copy
import math
import os
import pickle

from naslib.utils.utils import get_project_root
from ofa.utils import download_url

from ofa.imagenet_classification.elastic_nn.modules.dynamic_op import DynamicBatchNorm2d
import torch.nn.functional as F


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def construct_maps(keys):
    d = dict()
    keys = list(set(keys))
    for k in keys:
        if k not in d:
            d[k] = len(list(d.keys()))
    return d


ks_map = construct_maps(keys=(3, 5, 7))
ex_map = construct_maps(keys=(3, 4, 6))
dp_map = construct_maps(keys=(2, 3, 4))


def set_running_statistics(search_space, data_loader, device):
    """
    # This function adjusts the Batch norms in order to adapt to the validation Batchsize
    """
    bn_mean = {}
    bn_var = {}

    forward_search_space = copy.deepcopy(search_space)
    for name, m in forward_search_space.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_mean[name] = AverageMeter()
            bn_var[name] = AverageMeter()

            def new_forward(bn, mean_est, var_est):
                def lambda_forward(x):
                    batch_mean = (
                        x.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
                    )  # 1, C, 1, 1
                    batch_var = (x - batch_mean) * (x - batch_mean)
                    batch_var = (
                        batch_var.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
                    )

                    batch_mean = torch.squeeze(batch_mean)
                    batch_var = torch.squeeze(batch_var)

                    mean_est.update(batch_mean.data, x.size(0))
                    var_est.update(batch_var.data, x.size(0))

                    # bn forward using calculated mean & var
                    _feature_dim = batch_mean.size(0)
                    return F.batch_norm(
                        x,
                        batch_mean,
                        batch_var,
                        bn.weight[:_feature_dim],
                        bn.bias[:_feature_dim],
                        False,
                        0.0,
                        bn.eps,
                    )

                return lambda_forward

            m.forward = new_forward(m, bn_mean[name], bn_var[name])

    if len(bn_mean) == 0:
        # skip if there is no batch normalization layers in the network
        return

    forward_search_space.to(device)
    with torch.no_grad():
        DynamicBatchNorm2d.SET_RUNNING_STATISTICS = True
        for images, labels in data_loader:
            images = images.to(device)
            forward_search_space(images)
        DynamicBatchNorm2d.SET_RUNNING_STATISTICS = False

    for name, m in search_space.named_modules():
        if name in bn_mean and bn_mean[name].count > 0:
            feature_dim = bn_mean[name].avg.size(0)
            assert isinstance(m, nn.BatchNorm2d)
            m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
            m.running_var.data[:feature_dim].copy_(bn_var[name].avg)


def spec2feats(ks_list, ex_list, d_list, r):
    """This function converts a network config to a feature vector (128-D). One-hot-encoding."""
    start = 0
    end = 4
    for d in d_list:
        for j in range(start + d, end):
            ks_list[j] = 0
            ex_list[j] = 0
        start += 4
        end += 4

    # convert to onehot
    ks_onehot = [0 for _ in range(60)]
    ex_onehot = [0 for _ in range(60)]
    r_onehot = [0 for _ in range(8)]

    for i in range(20):
        start = i * 3
        if ks_list[i] != 0:
            ks_onehot[start + ks_map[ks_list[i]]] = 1
        if ex_list[i] != 0:
            ex_onehot[start + ex_map[ex_list[i]]] = 1

    r_onehot[(r - 112) // 16] = 1
    return torch.Tensor(ks_onehot + ex_onehot + r_onehot)


class AccuracyPredictor:
    """The pretrained Accuracy predictor from the original OFA implementation.
    """
    def __init__(self, pretrained=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = nn.Sequential(
            nn.Linear(128, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 1),
        )
        if pretrained:
            # load pretrained model
            fname = download_url(
                "https://hanlab.mit.edu/files/OnceForAll/tutorial/acc_predictor.pth"
            )
            self.model.load_state_dict(
                torch.load(fname, map_location=torch.device("cpu"))
            )
        self.model = self.model.to(self.device)

    @torch.no_grad()
    def predict_accuracy(self, population):
        all_feats = []
        for sample in population:
            ks_list = copy.deepcopy(sample["ks"])
            ex_list = copy.deepcopy(sample["e"])
            d_list = copy.deepcopy(sample["d"])
            r = copy.deepcopy(sample["r"])[0]
            feats = (
                spec2feats(ks_list, ex_list, d_list, r)
                .reshape(1, -1)
                .to(self.device)
            )
            all_feats.append(feats)
        all_feats = torch.cat(all_feats, 0)
        pred = self.model(all_feats).cpu()
        return pred


def build_val_transform(size):
    return transforms.Compose([
        transforms.Resize(int(math.ceil(size / 0.875))),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


class OFADatasetAPI(object):
    """This API serves as a provider of the different datasets, the queryable accuracy table and the pretrained
    accuracy predictor.
    """
    def __init__(self, dataset_path=None, data_path=None):
        if dataset_path:
            self.datasets_path = dataset_path
        else:
            root = get_project_root().parent
            self.datasets_path = os.path.join(root, ".datasets")
        if data_path:
            self.data_path = data_path
        else:
            root = get_project_root()
            self.data_path = os.path.join(root, "data")
        self.items = {}

        acc_pred = AccuracyPredictor()
        self.items["accuracy_predictor"] = acc_pred

        self.get_imagenet1k_subset()
        self.get_imagenet10k_subset()
        self.get_ofa_pickle()

    def get_imagenet1k_subset(self):
        path = os.path.join(self.datasets_path, "imagenet1k", "val")
        data_loader = DataLoader(
            datasets.ImageFolder(
                root=path,
                transform=build_val_transform(224)
            ),
            batch_size=250,  # test batch size
            num_workers=1,  # number of workers for the data loader
            pin_memory=True,
        )
        self.items["dataloader_test"] = data_loader

    def get_imagenet10k_subset(self):
        path = os.path.join(self.datasets_path, "imagenet10k", "val")
        data_loader = DataLoader(
            datasets.ImageFolder(
                root=path,
                transform=build_val_transform(224)
            ),
            batch_size=250,  # test batch size
            num_workers=1,  # number of workers for the data loader
            pin_memory=True,
        )
        self.items["dataloader_val"] = data_loader

    def get_ofa_pickle(self):
        self.data_path = os.path.join(get_project_root(), "data", "ofa.pickle")
        if os.path.exists(self.data_path):
            with open(self.data_path, "rb") as f:
                data = pickle.load(f)
            f.close()
        else:
            data = {}
        self.items['lut'] = data

    def close(self):
        """Saves the look-up-table."""
        data = self.items['lut']
        with open(self.data_path, "wb") as f:
            pickle.dump(data, f)
        f.close()

    def __getitem__(self, item):
        return self.items[item]
