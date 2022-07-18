from ofa.utils import profile
from ofa.model_zoo import ofa_net
from graph import OnceForAllSearchSpace
from torchvision.models import resnet50
from ofa.tutorial import FLOPsTable
from naslib.utils import measure_net_latency
from thop import profile
import torch


ss = OnceForAllSearchSpace()
of = ofa_net("ofa_mbv3_d234_e346_k357_w1.0")
input = torch.randn(8, 3, 224, 224)
macs, params = profile(ss, inputs=(input, ))
print(macs, params)

#
# print(profile(of, (1, 3, 224, 224)))
#
# model = resnet50()
# print(profile(model, (1, 3, 224, 224)))
lat, _ = measure_net_latency(ss, 'gpu64')
print(lat)
lat, _ = measure_net_latency(of, 'gpu64')
print(lat)
of.set_active_subnet(3, 3, 2)
of = of.get_active_subnet()
lat, _ = measure_net_latency(of, 'gpu64')
print(lat)

f_table = FLOPsTable(multiplier=1, batch_size=256, load_efficiency_table='local_lut.npy')
c = ss.get_active_conf_dict()
print(f_table.predict_efficiency(c))

ss.set_op_indices([0 for _ in range(20)] + [3 for _ in range(5)])
c = ss.get_active_conf_dict()
print(f_table.predict_efficiency(c))
lat, _ = measure_net_latency(ss, 'gpu64')
print(lat)
# for i in range(10):
#     ss.sample_random_architecture()
#     c = ss.get_active_conf_dict()
#     print(f_table.predict_efficiency(c))

