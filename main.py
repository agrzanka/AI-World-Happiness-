import numpy as np
import pandas as pd
import os
print(os.listdir("data"))
import statsmodels.formula.api as stats
from statsmodels.formula.api import ols
import sklearn
from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
#from plotly import optional_imports, tools, utils
#from plotly.exceptions import PlotlyError
#from ._plotlyjs_version import _plotlyjs_version_
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
happiness_2015 = pd.read_csv("data/2015.csv")
happiness_2015.columns = ['Country', 'Region', 'Happiness_Rank', 'Happiness_Score',
       'Standard Error', 'Economy', 'Family',
       'Health', 'Freedom', 'Trust',
       'Generosity', 'Dystopia_Residual']

columns_2015 = ['Region', 'Standard Error']
new_dropped_2015 = happiness_2015.drop(columns_2015, axis=1)
happiness_2016 =  pd.read_csv("data/2016.csv")
columns_2016 = ['Region', 'Lower Confidence Interval','Upper Confidence Interval' ]
dropped_2016 = happiness_2016.drop(columns_2016, axis=1)
dropped_2016.columns = ['Country', 'Happiness_Rank', 'Happiness_Score','Economy', 'Family',
       'Health', 'Freedom', 'Trust',
       'Generosity', 'Dystopia_Residual']
happiness_2017 =  pd.read_csv("data/2017.csv")
columns_2017 = ['Whisker.high','Whisker.low' ]
dropped_2017 = happiness_2017.drop(columns_2017, axis=1)
dropped_2017.columns = ['Country', 'Happiness_Rank', 'Happiness_Score','Economy', 'Family',
       'Health', 'Freedom', 'Trust',
       'Generosity', 'Dystopia_Residual']
frames = [new_dropped_2015, dropped_2016, dropped_2017]
happiness = pd.concat(frames)

happiness.head()