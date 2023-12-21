from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    Lars,
    Lasso,
    LassoLars,
    LinearRegression,
    MultiTaskElasticNet,
    MultiTaskLasso,
    OrthogonalMatchingPursuit,
    PassiveAggressiveRegressor,
    RANSACRegressor,
    Ridge,
    SGDRegressor,
    TheilSenRegressor,
)

# 加载波士顿房价数据集
data = fetch_openml(name='boston', version=1, as_frame=True)

# 获取特征和目标变量，并转换为NumPy数组
X = data.data.to_numpy()
y = data.target.to_numpy()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = [
    ARDRegression(),
    BayesianRidge(),
    ElasticNet(),
    Lars(),
    Lasso(),
    LassoLars(),
    LinearRegression(),
    OrthogonalMatchingPursuit(),
    PassiveAggressiveRegressor(),
    RANSACRegressor(),
    Ridge(),
    SGDRegressor(),
    TheilSenRegressor(),
]

for model in models:
    # 拟合模型
    model.fit(X_train, y_train)

    # 预测测试集
    y_pred = model.predict(X_test)

    # 计算均方误差
    mse = mean_squared_error(y_test, y_pred)
    print(f"{model.__class__.__name__}: Mean Squared Error: {mse}")