import torch
import torch.nn as nn
from torchvision import transforms, datasets

import copy
import math
import os
import pickle

from naslib.utils.utils import get_project_root
from ofa.utils import download_url


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


def spec2feats(ks_list, ex_list, d_list, r):
    # This function converts a network config to a feature vector (128-D).
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
    def __init__(self, dataset="imagenet", data_path=r'~/dataset/imagenet_1k'):
        self.ofa_data_path = ''
        self.items = {}
        data_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                root=os.path.join(data_path, 'val'),
                transform=build_val_transform(224)
            ),
            batch_size=250,  # test batch size
            # num_workers=16,  # number of workers for the data loader
            #  shuffle=True, originally in the code, but why?
            pin_memory=True,
            # drop_last=False,
        )
        self.items["data_loader"] = data_loader

        acc_pred = AccuracyPredictor()
        self.items["accuracy_predictor"] = acc_pred

        self.get_imagenet1k_pickle()

    def get_imagenet1k_pickle(self):
        self.ofa_data_path = os.path.join(get_project_root(), "data", "ofa_1k.pickle")
        if os.path.exists(self.ofa_data_path):
            with open(self.ofa_data_path, "rb") as f:
                data = pickle.load(f)
            f.close()
        else:
            data = {}
        self.items['ofa_data'] = data

    def close(self):
        data = self.items['ofa_data']
        with open(self.ofa_data_path, "wb") as f:
            pickle.dump(data, f)
        f.close()

    def __getitem__(self, item):
        return self.items[item]
