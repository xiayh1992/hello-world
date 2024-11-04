import numpy as np
from skopt import gp_minimize

np.random.seed(123)

# 使用黑盒函数。
def f(x):
    return (np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) *
            np.random.randn() * 0.1)

from skopt import Optimizer
opt = Optimizer([(-2.0, 2.0)])

for i in range(20):
    suggested = opt.ask()
    y = f(suggested)
    res = opt.tell(suggested, y)z
print("x*=%.2f f(x*)=%.2f" % (res.x[0], res.fun))
print(res.x_iters)
print(res.func_vals)

# ask-tell 接口，告诉以往历史，获取推荐值。"GP", "RF", "ET", "GBRT" or sklearn regressor, default: "GP"
opt = Optimizer([(-2.0, 2.0)], base_estimator = "RF")
x = [[0.8518212820929092], [0.8766012406190926], [-0.03552426626961047], [0.31877718809044087], [-1.4401969494784619], [-0.7033964264818355], [-1.0209628921843203], [0.527168070748202], [-0.23897128774369447], [-0.2885460366487216], [0.8674537062760881], [0.8136170500277524], [-1.9999879441715187], [1.1502538557586242], [-0.7489527857388156], [-0.6559726090374027], [-1.711998287744601], [1.6315941877440592], [0.8495387119676345], [-0.1824596817918942]]
y = [-6.05458547e-02,  2.22664068e-02,  7.79974860e-03,  1.24846197e-01,
 -4.14762245e-03, -2.59488268e-02,  1.62536846e-02,  5.82592922e-02,
 -2.07300216e-02,  4.21389150e-02, -1.04845796e-02,  6.40925188e-03,
 -3.34007696e-05,  6.40446963e-03,  1.51998060e-02,  1.24821476e-02,
  1.89629203e-04, -5.19809170e-05, -2.63528963e-02, -5.98541784e-02]
res = opt.tell(x, y)
print("x*=%.2f f(x*)=%.2f" % (res.x[0], res.fun))
print(res.func_vals)


# ask-tell 接口，分步骤告知历史数据，分步骤获取推荐值。
opt = Optimizer([(-2.0, 2.0)])
x = [[0.85], [1.2]]
y = [1.02, 0.34]
# start tell then ask
res = opt.tell(x, y)
suggested_x = opt.ask()
# next tell then ask
res = opt.tell(suggested, f(suggested_x))
suggested_x = opt.ask()


