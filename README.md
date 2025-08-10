# Real-World-Application-
This an exercise I did with a real world application. In this exercise, I was wokring to find the linear regression of a model in order to see what the effect the weather had on bike rentals. 
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from statsmodels.tsa.holtwinters import Holt, SimpleExpSmoothing
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, BayesianRidge
from sklearn.model_selection import train_test_split
from dmba import regressionSummary, exhaustive_search
from dmba import backward_elimination, forward_selection, stepwise_selection
from dmba import adjusted_r2_score, AIC_score, BIC_score

get_ipython().run_line_magic('matplotlib', 'inline')


data= pd.read_csv('day.csv') #downloading data set



data['dteday']= pd.date_range(start='1/1/2011',periods=len(data), freq='M')
data.index.freq='M' #explicitly stating frequency
data_series= data['cnt']


#defining metrics in order to be able distinguish which model is more effective

def calculate_metrics(actual, forecast):
    mae= mean_absolute_error(actual, forecast)
    mse= mean_squared_error(actual, forecast)
    rmse= np.sqrt(mse)
    mape= np.mean (np.abs((actual/forecast)/actual))*100
    return mape,mae,mse,rmse


#LinearRegression of a six month forecast for bike rentals

dteday= np.arange(len(data_series)).reshape(-1,1)
reg_model= LinearRegression().fit(dteday, data_series)
future_months= np.arange(len(data_series), len(data_series)+6).reshape(-1,1)
linear_regression= reg_model.predict(dteday)
linear_regression_future= reg_model.predict(future_months)


for i, prediction in enumerate(linear_regression_future, 1):
    print(f"Month {i} prediction:{prediction: .2f}")


#Decided Linear Regression was not the best model
#it seems unlikely that in the first 6 months of the next year
#6k bikes would be shared 
#also this market has trends not accounted for in linear regression

#exponential smoothing at smoothing level of 0.2

exp_smoothing_0_2_model= Holt(data_series, initialization_method= "estimated").fit(smoothing_level=0.2)
exp_smoothing_0_2_= exp_smoothing_0_2_model.fittedvalues
exp_smoothing_0_2_future= exp_smoothing_0_2_model.forecast(6)

for i, prediction in enumerate(exp_smoothing_0_2_future, 1):
    print(f"Month {i} prediction:{prediction: .2f}")

#exponential smoothing at smoothing level of 0.4

exp_smoothing_0_4_model= Holt(data_series, initialization_method= "estimated").fit(smoothing_level=0.4)
exp_smoothing_0_4_= exp_smoothing_0_4_model.fittedvalues
exp_smoothing_0_4_future= exp_smoothing_0_4_model.forecast(6)

for i, prediction in enumerate(exp_smoothing_0_4_future, 1):
    print(f"Month {i} prediction:{prediction: .2f}")

#exponential smoothing at smoothing level of 0.6

exp_smoothing_0_6_model= Holt(data_series, initialization_method= "estimated").fit(smoothing_level=0.6)
exp_smoothing_0_6_= exp_smoothing_0_6_model.fittedvalues
exp_smoothing_0_6_future= exp_smoothing_0_6_model.forecast(6)


for i, prediction in enumerate(exp_smoothing_0_6_future, 1):
    print(f"Month {i} prediction:{prediction: .2f}")


metrics = {
    'Exp Smoothing 0.2': calculate_metrics(data_series[1:],exp_smoothing_0_2_[1:]),
    'Exp Smoothing 0.4': calculate_metrics(data_series[1:],exp_smoothing_0_4_[1:]),
    'Exp Smoothing 0.6': calculate_metrics(data_series[1:],exp_smoothing_0_6_[1:]),
    'Linear Regression': calculate_metrics(data_series, linear_regression)
}


metrics_df= pd.DataFrame(metrics, index= ['MAPE','MAE','MSE','RMSE'])
metrics_df      

#metrics used to decide which model is best


#Here it is clear that Exp Smoothing 0.2 model has it has the lowest error rate
#Clear that Linear Regression was not the right choice; it has a high error rate 


#beginning of multiple linear regression to see how the environmental aspects affect how many bikes will be used
data= data.iloc[0:731]

predictors = ['temp','windspeed', 'hum']
outcome= 'cnt'

#set up validation with a 60/40 split and random seed to ensure model can be reused
X=data[predictors]
y=data[outcome]

train_X, valid_X, train_y, valid_y= train_test_split(X, y, test_size=0.4, random_state=1)

data_lr= LinearRegression()
data_lr.fit(train_X,train_y)


print(pd.DataFrame({'Predictor':X.columns, 'coeffecient':data_lr.coef_}))

#can see a clear increase of rentals when tempatrue increases
#can see a clear decrease of rentals when windspeed and humidity increases



#running regression summary 
regressionSummary(train_y, data_lr.predict(train_X))


#running the model to use 
data_lr_prediction= data_lr.predict(valid_X)

result= pd.DataFrame({'Predicted': data_lr_prediction, 'Actual':valid_y, 'Residual': valid_y - data_lr_prediction})
print(result.head(20))


regressionSummary(valid_y, data_lr.predict(valid_X))


#overall error rate is relatively close showing that the model is fairly accurate






