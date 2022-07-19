import pickle
from naslib.utils.utils import get_project_root
from naslib.search_spaces.core.query_metrics import Metric
import os
import pprint
from ofa_utils import OFADatasetAPI
from graph import OnceForAllSearchSpace
from naslib.predictors.utils import encodings
import time

# ss = OnceForAllSearchSpace()
# print(ss.encode())
# print(encodings.encode_adjacency_one_hot_ofa(ss))

# root = get_project_root()
# path = os.path.join(root, "data", "ofa.pickle")
#
# data = pickle.load(open(path, 'rb'))
# print(len(data))

api = OFADatasetAPI()

ss = OnceForAllSearchSpace()
ss.set_weights()

print(ss.query(metric=Metric.VAL_ACCURACY, dataset_api=api))
print(ss.query(metric=Metric.TEST_ACCURACY, dataset_api=api))

ss.sample_random_architecture()
start = time.time()
print(ss.query(metric=Metric.VAL_ACCURACY, dataset_api=api))
print(time.time() - start)
start = time.time()
print(ss.query(metric=Metric.TEST_ACCURACY, dataset_api=api))
print(time.time() - start)
