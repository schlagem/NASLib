from graph import OnceForAllSearchSpace
from ofa_utils import OFADatasetAPI
import time
from naslib.search_spaces.core.query_metrics import Metric


ss = OnceForAllSearchSpace()
ss.set_weights()

api = OFADatasetAPI()

start_time = time.time()
duration = 60 * 2
n = 1
time_spent = time.time() - start_time
while time_spent < duration:
    ss.sample_random_architecture()
    ss.query(Metric.TEST_ACCURACY, dataset_api=api)
    if n % 100 == 0:
        api.close()
    n += 1
    time_spent = time.time() - start_time
api.close()
