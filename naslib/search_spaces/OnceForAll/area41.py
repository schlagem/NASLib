from graph import *

ss = OnceForAllSearchSpace()
ss._set_weights()

print(ss.query(Metric.VAL_ACCURACY))
print(ss._eval_graph(Metric.VAL_ACCURACY))
