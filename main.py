import pandas as panda
from sklearn import linear_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def linearPredictionHappiness():
    df = panda.read_csv("data/2017.csv", sep=',')
    reg = linear_model.LinearRegression()
    reg.fit(df[['Freedom']], df[['Happiness.Score']])

    yPrediction = reg.predict(df[['Freedom']])

    plt.scatter(df[['Freedom']], df[['Happiness.Score']])
    plt.plot(df[['Freedom']], yPrediction, color='red')
    plt.show()

    print('Variance score: %.2f' % r2_score(df[['Happiness.Score']], yPrediction))
    print(reg.coef_)
    print ("==============================================================")
    print(reg.intercept_)
    print ("==============================================================")
    print(yPrediction)

def main():

    linearPredictionHappiness()




if __name__ == "__main__":
    main()