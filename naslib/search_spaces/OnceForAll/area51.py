from graph import OnceForAllSearchSpace
from ofa.tutorial import AccuracyPredictor
from ofa.tutorial import FLOPsTable, LatencyTable
from ofa.model_zoo import ofa_net
import torch
from ofa_utils import OFADatasetAPI
from naslib.utils.utils import get_project_root
import pickle
import os

# ofa_data_path = os.path.join(get_project_root(), "data", "ofa_1k.pickle")
# if os.path.exists(ofa_data_path):
#     with open(ofa_data_path, "rb") as f:
#         data = pickle.load(f)
#     f.close()
# else:
#     data = {}
# print(data)

a = [5, 0, 3, 3, 7, 3, 5, 2, 4, 7, 6, 8, 8, 1, 6, 7, 7, 8, 1, 5, 2, 2, 1, 2, 1]


ds_api = OFADatasetAPI()
ss = OnceForAllSearchSpace()
ss.set_weights()
ss.set_op_indices(a)
print(ss.evaluate(ds_api))

