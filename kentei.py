import math

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV, RidgeCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

def main():

    # サンプルデータを用意
    dataset = load_boston()
    # 標本データを取得
    X = dataset.data
    # 正解データを取得
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    print("X: %dx%dの行列" % (X.shape[0], X.shape[1]))
    print("y: %dのベクトル" % y.shape)

    # データの詳細
    print("目的変数：ボストンに建てられた住宅価格の中央値")
    X_columns = dataset.feature_names
    i = 0
    print("各カラムの構成")
    for column in X_columns:
        print("β%d：%s" % (i, column))
        i += 1

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
    plt.savefig('lr.pdf', transparent=True)
    plt.show()

    # alphaの候補
    alphaConf = 10 ** np.arange(-2, 3, 1, dtype=float)
    # ridge model

    # crossvalidation
    scaler = StandardScaler()
    crf = RidgeCV(alphas=10 ** np.arange(-2, 2, 0.1), cv=5)
    scaler.fit(X)
    crf.fit(scaler.transform(X), y)
    ridgeBestAlpha = crf.alpha_
    print('最適なλ：%f' % ridgeBestAlpha)


    print("\n ridge model")
    ridgeAlpha = []
    for alpha in alphaConf:
        if ridgeBestAlpha < alpha and alpha/10 < ridgeBestAlpha:
            ridgeAlpha.append(ridgeBestAlpha)
        ridgeAlpha.append(alpha)
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
    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient magnitude")
    plt.legend()
    plt.savefig('ridgeCoef.pdf', transparent=True)
    plt.show()
    alphaLog = []

    # lasso model

    # crossvalidation
    scaler = StandardScaler()
    clf = LassoCV(alphas=10 ** np.arange(-2, 2, 0.1), cv=5)
    scaler.fit(X)
    clf.fit(scaler.transform(X), y)
    lassoBestAlpha = clf.alpha_
    print('最適なλ：%f' % lassoBestAlpha)

    print("\n lasso model")
    lassoAlpha = []
    for alpha in alphaConf:
        if lassoBestAlpha < alpha and alpha/10 < lassoBestAlpha :
            lassoAlpha.append(lassoBestAlpha)
        lassoAlpha.append(alpha)
    lassoTrainSetScore = []
    lassoTestSetScore = []
    lassoBeta = []
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
        lassoBeta.append(lasso_pl.named_steps.lasso.coef_)
        for beta in lasso_pl.named_steps.lasso.coef_:
            print("β%d: %0.3f" % (i, beta))
            i += 1
    plt.legend()
    plt.xlabel("Coef index")
    plt.ylabel("Coef magnitude")
    plt.savefig('lassoCoef.pdf', transparent=True)
    plt.show()

    # Score Map
    for i in range(len(ridgeAlpha)):
        alphaLog.append(math.log10(ridgeAlpha[i]))
        print("α=%.4f: train=%.2f, test=%.2f" % (ridgeAlpha[i], ridgeTrainSetScore[i], ridgeTestSetScore[i]))
    alphaLog = []
    for i in range(len(lassoAlpha)):
        alphaLog.append(math.log10(lassoAlpha[i]))
        print("α=%.4f: train=%.2f, test=%.2f" % (lassoAlpha[i], lassoTrainSetScore[i], lassoTestSetScore[i]))
    plt.plot(alphaLog, ridgeTrainSetScore, label="ridge_train")
    plt.plot(alphaLog, ridgeTestSetScore, label="ridge_test")
    plt.plot(alphaLog, lassoTrainSetScore, label="lasso_train")
    plt.plot(alphaLog, lassoTestSetScore, label="lasso_test")
    plt.xlabel("log_10(α)")
    plt.plot([math.log10(ridgeBestAlpha), math.log10(ridgeBestAlpha)], [0, 1], "black", linestyle='dashed')
    plt.plot([math.log10(lassoBestAlpha), math.log10(lassoBestAlpha)], [0, 1], "black", linestyle='dashed')
    plt.legend()
    plt.savefig('score.pdf', transparent=True)
    plt.show()

    # solution path of lasso
    plt.plot(alphaLog, lassoBeta)
    plt.xlabel('log_10(α)')
    plt.ylabel('beta')
    plt.title('LASSO Path')
    plt.savefig('lassoPath.pdf', transparent=True)
    plt.show()


if __name__ == "__main__":
    main()
