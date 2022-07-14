from naslib.search_spaces.darts.graph import *
from naslib.search_spaces.nasbench201.graph import *
from graph import OnceForAllSearchSpace
from ofa.model_zoo import ofa_net

device = torch.device('cuda')

ss = OnceForAllSearchSpace()
ss.set_weights()
ss_p = ss.parameters()

net_id = "ofa_mbv3_d234_e346_k357_w1.0"
ofa_network = ofa_net(net_id, pretrained=True)
ofa_p = ofa_network.parameters()


