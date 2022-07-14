import copy
import time
import torch


def measure_net_latency(
        net, l_type="gpu8", fast=True, input_shape=(3, 224, 224), clean=True
):
    # return `ms`
    if "gpu" in l_type:
        l_type, batch_size = l_type[:3], int(l_type[3:])
    else:
        batch_size = 1

    data_shape = [batch_size] + list(input_shape)
    if l_type == "cpu":
        if fast:
            n_warmup = 5
            n_sample = 10
        else:
            n_warmup = 50
            n_sample = 50
        if get_net_device(net) != torch.device("cpu"):
            if not clean:
                print("move net to cpu for measuring cpu latency")
            net = copy.deepcopy(net).to(torch.device("cpu"))
    elif l_type == "gpu":
        if fast:
            n_warmup = 5
            n_sample = 10
        else:
            n_warmup = 50
            n_sample = 50
        if get_net_device(net) != torch.device("cuda"):
            if not clean:
                print("move net to gpu for measuring gpu latency")
            net = net.to(torch.device("cuda"))
    else:
        raise NotImplementedError
    images = torch.zeros(data_shape, device=get_net_device(net))

    measured_latency = {"warmup": [], "sample": []}
    net.eval()
    with torch.no_grad():
        for i in range(n_warmup):
            inner_start_time = time.time()
            net(images)
            used_time = (time.time() - inner_start_time) * 1e3  # ms
            measured_latency["warmup"].append(used_time)
            if not clean:
                print("Warmup %d: %.3f" % (i, used_time))
        outer_start_time = time.time()
        for i in range(n_sample):
            net(images)
        total_time = (time.time() - outer_start_time) * 1e3  # ms
        measured_latency["sample"].append((total_time, n_sample))
    return total_time / n_sample, measured_latency


def get_net_device(net):
    return next(net.parameters()).device


def count_parameters(net):
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_params
