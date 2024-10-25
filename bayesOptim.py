from bayes_opt import BayesianOptimization
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

# 定义目标函数
def ridge_objective(alpha, epsilon):
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1)
    model = Ridge(alpha=alpha, fit_intercept=True)
    scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
    return scores.mean()

# 定义优化器
optimizer = BayesianOptimization(
    f=ridge_objective,
    pbounds={"alpha": (0.1, 10), "epsilon": (0.01, 1)},
    random_state=1,
)

# 进行优化
optimizer.maximize(init_points=2, n_iter=20)

# 获取最佳参数组合
best_params = optimizer.max['params']
print("Best parameters:", best_params)
print("-----------------------")
print(ridge_objective(0.1011, 0.0734))