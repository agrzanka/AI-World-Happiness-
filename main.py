import pandas as panda
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import collections
import numpy as np

from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


def linearPredictionHappiness():
    df = panda.read_csv("data/2017.csv", sep=',')

    # ============= linear regression: single variable ==================================

    regFreedom = linear_model.LinearRegression()
    regFreedom.fit(df[['Freedom']], df[['Happiness.Score']])
    yPrediction = regFreedom.predict(df[['Freedom']])
    plt.xlabel('Freedom')
    plt.ylabel('Happiness Score')
    plt.scatter(df[['Freedom']], df[['Happiness.Score']])
    plt.plot(df[['Freedom']], yPrediction, color='red')
    plt.show()


    regHealth=linear_model.LinearRegression()
    regHealth.fit(df[['Health..Life.Expectancy.']], df[['Happiness.Score']])
    yPredictionHealth = regHealth.predict(df[['Health..Life.Expectancy.']])
    plt.xlabel('Health -Life Expectancy')
    plt.ylabel('Happiness Score')
    plt.scatter(df[['Health..Life.Expectancy.']], df[['Happiness.Score']])
    plt.plot(df[['Health..Life.Expectancy.']], yPredictionHealth , color='red')
    plt.show()

    regEconomy = linear_model.LinearRegression()
    regEconomy.fit(df[['Economy..GDP.per.Capita.']], df[['Happiness.Score']])
    yPredictionEconomy = regEconomy.predict(df[['Economy..GDP.per.Capita.']])
    plt.xlabel('Economy -GDP per Capita')
    plt.ylabel('Happiness Score')
    plt.scatter(df[['Economy..GDP.per.Capita.']], df[['Happiness.Score']])
    plt.plot(df[['Economy..GDP.per.Capita.']], yPredictionEconomy, color='red')
    plt.show()

    regFamily = linear_model.LinearRegression()
    regFamily.fit(df[['Family']], df[['Happiness.Score']])
    yPredictionFamily = regFamily.predict(df[['Family']])
    plt.xlabel('Family')
    plt.ylabel('Happiness Score')
    plt.scatter(df[['Family']], df[['Happiness.Score']])
    plt.plot(df[['Family']], yPredictionFamily, color='red')
    plt.show()

    regGenerosity = linear_model.LinearRegression()
    regGenerosity.fit(df[['Generosity']], df[['Happiness.Score']])
    yPredictionGenerosity = regGenerosity.predict(df[['Generosity']])
    plt.xlabel('Generosity')
    plt.ylabel('Happiness Score')
    plt.scatter(df[['Generosity']], df[['Happiness.Score']])
    plt.plot(df[['Generosity']], yPredictionGenerosity, color='red')
    plt.show()

    regTrust = linear_model.LinearRegression()
    regTrust.fit(df[['Trust..Government.Corruption.']], df[['Happiness.Score']])
    yPredictionTrust = regTrust.predict(df[['Trust..Government.Corruption.']])
    plt.xlabel('Trust -Government corruption')
    plt.ylabel('Happiness Score')
    plt.scatter(df[['Trust..Government.Corruption.']], df[['Happiness.Score']])
    plt.plot(df[['Trust..Government.Corruption.']], yPredictionTrust, color='red')
    plt.show()

    regDystopia = linear_model.LinearRegression()
    regDystopia.fit(df[['Dystopia.Residual']], df[['Happiness.Score']])
    yPredictionDystopia = regDystopia.predict(df[['Dystopia.Residual']])
    plt.xlabel('Dystopia')
    plt.ylabel('Happiness Score')
    plt.scatter(df[['Dystopia.Residual']], df[['Happiness.Score']])
    plt.plot(df[['Dystopia.Residual']], yPredictionDystopia, color='red')
    plt.show()

# linear regression for multiple variable - only to choose which of them are the most important
    print ("==============================================================")
    regMultiple2 = linear_model.LinearRegression()
    regMultiple2.fit(df[['Freedom','Health..Life.Expectancy.', 'Economy..GDP.per.Capita.', 'Family', 'Generosity', 'Trust..Government.Corruption.', 'Dystopia.Residual']], df[['Happiness.Score']])
    print('ALL:\ncoef: FREEDOM, HEALTH, ECONOMY, FAMILY, GENEROSITY, TRUST, DYSTOPIA')
    print(regMultiple2.coef_)
    print ("intercept:")
    print(regMultiple2.intercept_)
    print ("==============================================================")

    print('HEALTH:\ncoef:')
    print(regHealth.coef_)
    print ("intercept:")
    print(regHealth.intercept_)
    print ("==============================================================")
    print('Economy:\ncoef:')
    print(regEconomy.coef_)
    print ("intercept:")
    print(regEconomy.intercept_)
    print ("==============================================================")
    print('HEALTH:\ncoef:')
    print(regFamily.coef_)
    print ("intercept:")
    print(regFamily.intercept_)
    print ("==============================================================")


    # ============= linear regression: multiple variable ==================================
    regMultiple = linear_model.LinearRegression()
    regMultiple.fit(df[['Health..Life.Expectancy.','Economy..GDP.per.Capita.','Family' ]], df[['Happiness.Score']])
    print('MULTIPLE 1:\ncoef: HEALTH, ECONOMY, FAMILY')
    print(regMultiple.coef_)
    print ("intercept:")
    print(regMultiple.intercept_)

    print ("==============================================================")
    regMultiple3 = linear_model.LinearRegression()
    regMultiple3.fit(df[['Freedom', 'Economy..GDP.per.Capita.', 'Generosity']], df[['Happiness.Score']])
    print('MULTIPLE 2:\ncoef: FREEDOM, ECONOMY, GENEROSITY')
    print(regMultiple3.coef_)
    print ("intercept:")
    print(regMultiple3.intercept_)


def f(x):
    return 1/(1+np.exp(-x))

def logisticPredictionHappiness():
    df1 = panda.read_csv("data/2017.2.csv", sep=',')

    plt.scatter(df1[['Freedom']], df1[['Binary.Happiness.Score']])
    plt.xlabel('Freedom')
    plt.ylabel('Binary Happiness Score')
    plt.show()

    X_train, X_test, y_train, y_test=train_test_split(df1[['Freedom']], df1[['Binary.Happiness.Score']], test_size=0.01)
    model=LogisticRegression()
    model.fit(X_train,y_train.values.ravel())
    print("data used to train our model:")
    print(X_train)
    print("\nvalues to perform our test on ")
    print (X_test)
    model.predict_proba(X_test)
    print("\nprobability:")
    print(model.predict_proba(X_test))
    print("\nmodel score")
    print(model.score(X_test, y_test))


    plt.scatter(df1[['Health..Life.Expectancy.']], df1[['Binary.Happiness.Score']])
    plt.xlabel('Health')
    plt.ylabel('Binary Happiness Score')
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(df1[['Health..Life.Expectancy.']],
                                                        df1[['Binary.Happiness.Score']], test_size=0.01)
    model = LogisticRegression()
    model.fit(X_train, y_train.values.ravel())
    print("data used to train our model:")
    print(X_train)
    print("\nvalues to perform our test on ")
    print (X_test)
    model.predict_proba(X_test)
    print("\nprobability:")
    print(model.predict_proba(X_test))
    print("\nmodel score")
    print(model.score(X_test, y_test))


    plt.scatter(df1[['Economy..GDP.per.Capita.']], df1[['Binary.Happiness.Score']])
    plt.xlabel('Economy')
    plt.ylabel('Binary Happiness Score')
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(df1[['Economy..GDP.per.Capita.']],
                                                        df1[['Binary.Happiness.Score']], test_size=0.01)
    model = LogisticRegression()
    model.fit(X_train, y_train.values.ravel())
    print("data used to train our model:")
    print(X_train)
    print("\nvalues to perform our test on ")
    print (X_test)
    model.predict_proba(X_test)
    print("\nprobability:")
    print(model.predict_proba(X_test))
    print("\nmodel score")
    print(model.score(X_test, y_test))


    plt.scatter(df1[['Family']], df1[['Binary.Happiness.Score']])
    plt.xlabel('Family')
    plt.ylabel('Binary Happiness Score')
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(df1[['Family']],
                                                        df1[['Binary.Happiness.Score']], test_size=0.01)
    model = LogisticRegression()
    model.fit(X_train, y_train.values.ravel())
    print("data used to train our model:")
    print(X_train)
    print("\nvalues to perform our test on ")
    print (X_test)
    model.predict_proba(X_test)
    print("\nprobability:")
    print(model.predict_proba(X_test))
    print("\nmodel score")
    print(model.score(X_test, y_test))


    plt.scatter(df1[['Generosity']], df1[['Binary.Happiness.Score']])
    plt.xlabel('Generosity')
    plt.ylabel('Binary Happiness Score')
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(df1[['Generosity']],
                                                        df1[['Binary.Happiness.Score']], test_size=0.01)
    model = LogisticRegression()
    model.fit(X_train, y_train.values.ravel())
    print("data used to train our model:")
    print(X_train)
    print("\nvalues to perform our test on ")
    print (X_test)
    model.predict_proba(X_test)
    print("\nprobability:")
    print(model.predict_proba(X_test))
    print("\nmodel score")
    print(model.score(X_test, y_test))


    plt.scatter(df1[['Trust..Government.Corruption.']], df1[['Binary.Happiness.Score']])
    plt.xlabel('Trust')
    plt.ylabel('Binary Happiness Score')
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(df1[['Trust..Government.Corruption.']],
                                                        df1[['Binary.Happiness.Score']], test_size=0.01)
    model = LogisticRegression()
    model.fit(X_train, y_train.values.ravel())
    print("data used to train our model:")
    print(X_train)
    print("\nvalues to perform our test on ")
    print (X_test)
    model.predict_proba(X_test)
    print("\nprobability:")
    print(model.predict_proba(X_test))
    print("\nmodel score")
    print(model.score(X_test, y_test))


    plt.scatter(df1[['Dystopia.Residual']], df1[['Binary.Happiness.Score']])
    plt.xlabel('Dystopia')
    plt.ylabel('Binary Happiness Score')
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(df1[['Dystopia.Residual']],
                                                        df1[['Binary.Happiness.Score']], test_size=0.01)
    model = LogisticRegression()
    model.fit(X_train, y_train.values.ravel())
    print("data used to train our model:")
    print(X_train)
    print("\nvalues to perform our test on ")
    print (X_test)
    model.predict_proba(X_test)
    print("\nprobability:")
    print(model.predict_proba(X_test))
    print("\nmodel score")
    print(model.score(X_test,y_test))


def main():

    linearPredictionHappiness()
    print('=============================== LOGISTIC REGRESSION ==========================================')
    logisticPredictionHappiness()


if __name__ == "__main__":
    main()