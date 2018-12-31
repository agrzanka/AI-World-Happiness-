import pandas as panda
from sklearn import linear_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


def linearPredictionHappiness():
    df = panda.read_csv("data/2017.csv", sep=',')

    # ============= linear regresion: single variable ==================================

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


    #============= linear regresion: multiple variable ==================================




def main():

    linearPredictionHappiness()


if __name__ == "__main__":
    main()