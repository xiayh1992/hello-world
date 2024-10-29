# from bayes_opt import BayesianOptimization
# from sklearn.datasets import make_regression
# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import Ridge

# # 定义目标函数
# def ridge_objective(alpha, epsilon):
#     X, y = make_regression(n_samples=100, n_features=10, noise=0.1)
#     model = Ridge(alpha=alpha, fit_intercept=True)
#     scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
#     return scores.mean()

# # 定义优化器
# optimizer = BayesianOptimization(
#     f=ridge_objective,
#     pbounds={"alpha": (0.1, 10), "epsilon": (0.01, 1)},
#     random_state=1,
# )

# # 进行优化
# optimizer.maximize(init_points=2, n_iter=20)

# # 获取最佳参数组合
# best_params = optimizer.max['params']
# print("Best parameters:", best_params)
# print("-----------------------")
# print(ridge_objective(0.1011, 0.0734))





import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt

# 定义目标函数
def objective(x):
    x, y = x[0], x[1]
    return (x - 5) ** 2 + (y - 5) ** 2

# 初始采样点
initial_points = np.array([
    [0, 0],
    [1, 1],
    [2, 2],
    [3, 3],
    [4, 4],
    [6, 6],
    [7, 7],
    [8, 8],
    [9, 9],
    [10, 10]
])
initial_values = np.array([objective(x) for x in initial_points])

# 定义搜索空间
search_space = [Real(0, 10), Real(0, 10)]

# 初始化数据
X = initial_points
Y = initial_values

# 迭代次数
n_iterations = 10

for i in range(n_iterations):
    # 使用已有数据进行贝叶斯优化
    res = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=1,  # 每次只做一次迭代
        n_random_starts=0,  # 不再进行随机采样
        acq_func="EI",  # 采集函数类型
        x0=X,  # 已有采样点
        y0=Y,  # 已有目标函数值
        random_state=1  # 随机种子
    )
    
    # 更新数据集
    new_x = res.x_iters[-1].reshape(1, -1)
    new_y = objective(new_x)
    
    X = np.vstack([X, new_x])
    Y = np.append(Y, new_y)
    
    # 打印当前迭代的信息
    print(f"Iteration {i+1}: Next sample point: {new_x}, Objective value: {new_y}")

# 输出最终结果
best_index = np.argmin(Y)
best_x = X[best_index]
best_y = Y[best_index]
print(f"Best point found: {best_x}, Best objective value: {best_y}")

# 可视化
# 创建网格
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
xx, yy = np.meshgrid(x, y)
Z = objective(np.vstack((xx.ravel(), yy.ravel())).T).reshape(xx.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(xx, yy, Z, cmap='viridis', alpha=0.5)
ax.scatter(X[:, 0], X[:, 1], Y, color='r', marker='x')
ax.scatter(best_x[0], best_x[1], best_y, color='g', marker='o')

plt.show()