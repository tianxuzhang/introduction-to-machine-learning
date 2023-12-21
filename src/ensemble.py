from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import pydotplus
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 加载数据
def random_forest_classifier():
    data = load_iris()
    X = data.data
    y = data.target

    # 划分训练集和测试集
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建随机森林分类器
    model = RandomForestClassifier()

    # 训练模型
    model.fit(X_train, y_train)

    # 可视化决策树（第一棵树）
    estimator = model.estimators_[0]
    dot_data = export_graphviz(estimator, out_file=None, feature_names=data.feature_names,
                               class_names=data.target_names, filled=True, rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)

    # 保存为图像文件
    graph.write_png('random_forest_tree.png')

    # 使用matplotlib显示图像
    plt.figure(figsize=(10, 10))
    plt.imshow(plt.imread('random_forest_tree.png'))
    plt.axis('off')
    plt.show()

def random_forest_regressor():
    # 加载数据
    boston = fetch_openml(name='boston', version=1)
    X = boston.data
    y = boston.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建随机森林回归模型
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)

    # 训练模型
    rf_reg.fit(X_train, y_train)
    # print(rf_reg.estimators_)
    # 预测测试集
    y_pred = rf_reg.predict(X_test)

    # 计算均方误差
    mse = mean_squared_error(y_test, y_pred)
    print('Mean Squared Error:', mse)

def bagging_classifier():
    # 加载数据集
    data = load_iris()
    X = data.data
    y = data.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 实例化决策树分类器
    base_classifier = DecisionTreeClassifier()

    # 实例化Bagging分类器
    bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10)

    # 拟合模型
    bagging_classifier.fit(X_train, y_train)

    # 预测测试集
    y_pred = bagging_classifier.predict(X_test)
    print(bagging_classifier.estimators_)
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

def bagging_regressor():
    # 加载数据集
    data = fetch_openml(name='boston', version=1)
    X = data.data
    y = data.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 实例化决策树回归器
    base_regressor = DecisionTreeRegressor()

    # 实例化Bagging回归器
    bagging_regressor = BaggingRegressor(base_regressor, n_estimators=10)

    # 拟合模型
    bagging_regressor.fit(X_train, y_train)

    # 预测测试集
    y_pred = bagging_regressor.predict(X_test)
    print(bagging_regressor.estimators_)
    # 计算均方误差
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

def ada_boost_classifier():
    # 加载数据集
    data = load_iris()
    X = data.data
    y = data.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 实例化决策树分类器
    base_classifier = DecisionTreeClassifier()

    # 实例化AdaBoost分类器
    adaboost_classifier = AdaBoostClassifier(base_classifier, n_estimators=10)

    # 拟合模型
    adaboost_classifier.fit(X_train, y_train)

    # 预测测试集
    y_pred = adaboost_classifier.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)


def extra_trees_classifier():
    # 加载数据集
    data = load_iris()
    X = data.data
    y = data.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 实例化ExtraTrees分类器
    extra_trees_classifier = ExtraTreesClassifier(n_estimators=100)

    # 拟合模型
    extra_trees_classifier.fit(X_train, y_train)

    # 预测测试集
    y_pred = extra_trees_classifier.predict(X_test)
    print(extra_trees_classifier.estimators_)
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

# no
def voting_classifier():
    # 加载数据集
    data = load_iris()
    X = data.data
    y = data.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 实例化各个基分类器
    logistic_regression = LogisticRegression()
    svm_classifier = SVC(probability=True)
    decision_tree = DecisionTreeClassifier()

    # 实例化Voting分类器
    voting_classifier = VotingClassifier(estimators=[
        ('lr', logistic_regression),
        ('svm', svm_classifier),
        ('dt', decision_tree)
    ])

    # 拟合模型
    voting_classifier.fit(X_train, y_train)
    print(voting_classifier.estimators_)
    # 预测测试集
    y_pred = voting_classifier.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

def extra_trees_regressor():

    # 加载数据集
    data = fetch_openml(name='boston', version=1)
    X = data.data
    y = data.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 实例化ExtraTrees回归器
    extra_trees_regressor = ExtraTreesRegressor(n_estimators=100)

    # 拟合模型
    extra_trees_regressor.fit(X_train, y_train)
    print(extra_trees_regressor.estimators_)
    # 预测测试集
    y_pred = extra_trees_regressor.predict(X_test)

    # 计算均方误差
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

# no
def voting_regressor():
    # 加载数据集
    data = fetch_openml(name='boston', version=1)
    X = data.data
    y = data.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 实例化各个基回归器
    linear_regression = LinearRegression()
    knn_regressor = KNeighborsRegressor()
    decision_tree = DecisionTreeRegressor()

    # 实例化Voting回归器
    voting_regressor = VotingRegressor(estimators=[
        ('lr', linear_regression),
        ('knn', knn_regressor),
        ('dt', decision_tree)
    ])

    # 拟合模型
    voting_regressor.fit(X_train, y_train)
    print(voting_regressor.estimators_)
    # 预测测试集
    y_pred = voting_regressor.predict(X_test)

    # 计算均方误差
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

def linear_regression():

    # 加载数据集
    data = fetch_openml(name='boston', version=1)
    X = data.data
    y = data.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 实例化线性回归模型
    linear_regression = LinearRegression()

    # 拟合模型
    linear_regression.fit(X_train, y_train)

    # 预测测试集
    y_pred = linear_regression.predict(X_test)

    # 计算均方误差
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

def k_neighbors_regressor():
    data = fetch_openml(name='boston', version=1)
    X = data.data
    y = data.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 实例化K近邻回归器
    knn_regressor = KNeighborsRegressor(n_neighbors=5)

    # 拟合模型
    knn_regressor.fit(X_train, y_train)

    # 预测测试集
    y_pred = knn_regressor.predict(X_test)

    # 计算均方误差
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
# random_forest_regressor()
# random_forest_classifier()
# bagging_classifier()
# bagging_regressor()
# ada_boost_classifier()
# extra_trees_classifier()
# voting_classifier()
# extra_trees_regressor()
# voting_regressor()
linear_regression()
# k_neighbors_regressor()