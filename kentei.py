import csv

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

import mglearn
from sklearn.model_selection import train_test_split

def main():
    # 大きいサイズのサンプルデータ
    X, y = mglearn.datasets.load_extended_boston()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    print("X: %dx%dの行列" % (X.shape[0], X.shape[1]))
    print("y: %dのベクトル" % y.shape)

    # 重み
    poly = PolynomialFeatures(degree=X.shape[1])

    # least squares model
    print("\n least squares model")
    lr = LinearRegression().fit(X_train, y_train)
    print("Train set score: {:.2f}".format(lr.score(X_train, y_train)))
    print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))
    plt.plot(lr.coef_)
    lr_pl = Pipeline([("poly_10", poly), ("lr", lr)])
    print("バイアス: %f" % lr_pl.named_steps.lr.intercept_)
    i = 0
    for beta in lr_pl.named_steps.lr.coef_:
        print("β%d: %0.3f" % (i, beta))
        i += 1
    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient magnitude")
    plt.show()

    # ridge model
    print("\n ridge model")
    ridgeAlpha = [0.1, 1, 3, 7, 10]
    ridgeTrainSetScore = []
    ridgeTestSetScore = []
    for alpha in ridgeAlpha:
        ridge = Ridge(alpha=alpha).fit(X_train, y_train)
        print("\n alpha={}".format(str(alpha)))
        print("Train set score: {:.2f}".format(ridge.score(X_train, y_train)))
        ridgeTrainSetScore.append(ridge.score(X_train, y_train))
        print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))
        ridgeTestSetScore.append(ridge.score(X_test, y_test))
        plt.plot(ridge.coef_, label="alpha:{}".format(str(alpha)))
        ridge_pl = Pipeline([("poly_10", poly), ("ridge", ridge)])
        print("バイアス: %f" % ridge_pl.named_steps.ridge.intercept_)
        i = 0
        for beta in ridge_pl.named_steps.ridge.coef_:
            print("β%d: %0.3f" % (i, beta))
            i += 1
    for i in range(len(ridgeAlpha)):
        print("α=%.2f: train=%.2f, test=%.2f" % (ridgeAlpha[i], ridgeTrainSetScore[i], ridgeTestSetScore[i]))
    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient magnitude")
    plt.legend()
    plt.show()
    mglearn.plots.plot_ridge_n_samples()
    plt.show()

    # lasso model
    print("\n lasso model")
    lassoAlpha = [0.02, 0.5, 1]
    lassoTrainSetScore = []
    lassoTestSetScore = []
    for alpha in lassoAlpha:
        lasso = Lasso(alpha=alpha).fit(X_train, y_train)
        print("\n alpha={}".format(str(alpha)))
        print("Train set score: {:.2f}".format(lasso.score(X_train, y_train)))
        lassoTrainSetScore.append(lasso.score(X_train, y_train))
        print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
        lassoTestSetScore.append(lasso.score(X_test, y_test))
        print("Number of features used:{}".format(np.sum(lasso.coef_ != 0)))
        plt.plot(lasso.coef_, label="alpha:{}".format(str(alpha)))
        lasso_pl = Pipeline([("poly_10", poly), ("lasso", lasso)])
        print("バイアス: %f" % lasso_pl.named_steps.lasso.intercept_)
        i = 0
        for beta in lasso_pl.named_steps.lasso.coef_:
            print("β%d: %0.3f" % (i, beta))
            i += 1
    for i in range(len(lassoAlpha)):
        print("α=%.2f: train=%.2f, test=%.2f" % (lassoAlpha[i], lassoTrainSetScore[i], lassoTestSetScore[i]))
    plt.legend()
    plt.xlabel("Coef index")
    plt.ylabel("Coef magnitude")
    plt.show()


if __name__ == "__main__":
    main()
