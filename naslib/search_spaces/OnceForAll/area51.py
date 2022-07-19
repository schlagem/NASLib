import pickle
from naslib.utils.utils import get_project_root
import os
import pprint

root = get_project_root()
path = os.path.join(root, "data", "ofa.pickle")

data = pickle.load(open(path, 'rb'))
print(len(data))
