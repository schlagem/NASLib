from graph import OnceForAllSearchSpace
import numpy as np
from ofa.model_zoo import ofa_net
from naslib.utils import measure_net_latency

net = ofa_net("ofa_mbv3_d234_e346_k357_w1.0")




# ss = OnceForAllSearchSpace()
# up = ss.get_model_size()
# # print(measure_net_latency(ss, l_type="cpu", fast=0, input_shape=(3, 224, 224), clean=1))
#
# ss.set_op_indices([0 for _ in range(20)] + [3 for _ in range(5)])
# low = ss.get_model_size()
# print(low)
# # print(measure_net_latency(ss, l_type="cpu", fast=0, input_shape=(3, 224, 224), clean=1))
#
# print(np.quantile([low, up], 0.25))
# print(np.quantile([low, up], 0.5))
# print(np.quantile([low, up], 0.75))
# print(np.quantile([low, up], 1))
