# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 20:05:38 2019

@author: karthick
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA, ARMA
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson

###############################################################################
""" Reading the Dataset and indexing with Datetime """

df = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\ML\\sabree travels\\dataset.csv")
df = df.drop("Forecasted Bookings",axis=1)
df.ArrivalDate = pd.to_datetime(df.ArrivalDate,format='%d-%m-%Y')
df.index = df.ArrivalDate

""" Training and Test dataset """
df_training = df.loc["2014",:]
df_testing = df.loc["2015",:]

""" Splitting for a particular plane and class """
df_train_697 = df_training[df_training.FlightID==697]
df_test_697 = df_testing[df_testing.FlightID==697]
df_train_697_eco = df_train_697[df_train_697.Cabin=="Economy Cabin"]
df_test_697_eco = df_test_697[df_test_697.Cabin=="Economy Cabin"]

train_series = df_train_697_eco.AverageFarePerBooking.values.reshape(df_train_697_eco.shape[0],1)
test_series = df_test_697_eco.AverageFarePerBooking.values.reshape(df_test_697_eco.shape[0],1)

###############################################################################
""" Preliminary Visualisation """

plt.figure(1)
plt.subplot(211)
df_train_697_eco.AverageFarePerBooking.hist()
plt.subplot(212)
df_test_697_eco.AverageFarePerBooking.plot(kind='kde')
plt.show()

"""
Close to normal distribution
Kinda skewed
"""
df_train_697_eco.AverageFarePerBooking.plot()
plt.show()

decomposed = seasonal_decompose(df_train_697_eco.AverageFarePerBooking, model='additive',freq = 7)
decomposed.plot()
plt.show()

###############################################################################
""" Check for Stationary process
'''Augmented dickey fuller test
Null hypothesis : There is a unit root in the time series sample
Alternative Hypothesis : The series doesn't have a unit root., The series is stationary.''' """

ds_series = df_train_697_eco.AverageFarePerBooking.diff(7).dropna()
adf_results = adfuller(ds_series)
print(adf_results)

diff_series = df_train_697_eco.AverageFarePerBooking.diff().dropna()
adf_results = adfuller(diff_series)
print(adf_results)

ds_diff_series = df_train_697_eco.AverageFarePerBooking.diff(7).dropna().diff().dropna()
adf_results = adfuller(ds_diff_series)
print(adf_results)

"""saving the sationery series"""
ds_diff_series.to_csv('stationary_series.csv')
###############################################################################
""" PACF and ACF plots"""

plot_acf(ds_diff_series,lags = 50)
plot_pacf(ds_diff_series, lags = 50)
plt.show()

"""In partial auto correlation plot : lag 1,2,3,7,14,21 are significant (AR params)
In auto correlation plot: lag 1 and 7 are significant"""

"Will be useful for deciding level of grid search"

###############################################################################
"""Grid search for parameters using ARMA model with 1 diff and 7 seasonal-lag"""

aic_array = []
bic_array = []

for p in range(0,15):
    for q in range(0,15):
        order = (p,q)
        try:
            model = ARMA(ds_diff_series,order).fit()
            aic_array.append(((p,q),round(model.aic)))
            bic_array.append(((p,q),round(model.bic)))
            print('ARIMA%s AIC=%.3f' % (order,round(model.aic)))
        except:
            continue
        
indices = []; aics =[]; bics = []

for i in range(len(aic_array)):
    indices.append(aic_array[i][0])
    aics.append(aic_array[i][1])
    bics.append(bic_array[i][1])
    
print (" model according to aic ", indices[np.array(aics).argmin()])
print (" model according to bic ", indices[np.array(bics).argmin()])
################################################################################
"""Validation using Rolling window average """

history = list(df_train_697_eco.AverageFarePerBooking)
predictions = list()
test =  list(df_test_697_eco.AverageFarePerBooking)

for i in range(len(test)):
    model = sm.tsa.statespace.SARIMAX(history, order=(1,1,7),seasonal_order=(0,0,0,7),enforce_invertibility=False).fit()
    y_pred = model.forecast()[0]
    predictions.append(y_pred)
    history.append(test[i])
    print("Predicted = %.3f, Expected = %.3f"%(y_pred,test[i]))
    
print('RMSE: %.3f' % sqrt(mean_squared_error(test,predictions)))

""" RMSE for 30 seasonal (monthly) : 45.13"""
""" RMSE for 7 seasonal (weekly) : 28.914 """

plt.plot(test, color = 'blue', label ="True Value")
plt.plot(predictions,color = 'red', label ="Predictions")
plt.legend()
plt.show()

###############################################################################
"""Residuals Analysis"""

residuals = np.array(test)-np.array(predictions)
plt.hist(residuals)
plt.show()
"""seems to have a small bias thou ( close to centred around zero )"""

"""Residuals are not correlated """
plot_acf(residuals,lags = 50)
plot_pacf(residuals, lags = 50)
plt.show()

"""Residuals are stationary """
residuals_adf = adfuller(residuals)
print(adf_results)

"""Residuals are not normal"""

qqplot(residuals)

def normality_test(series):
    stat, p = shapiro(series)
    print('Shapiro test Statistics=%.3f, p=%.3f' % (stat, p))
    stat, p = normaltest(series)
    if p > 0.05:
    	print('Sample looks Gaussian (fail to reject H0)')
    else:
    	print('Sample does not look Gaussian (reject H0)')
    print('D’Agostino’s K^2 test Statistics=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
    	print('Sample looks Gaussian (fail to reject H0)')
    else:
    	print('Sample does not look Gaussian (reject H0)')
    result = anderson(series)
    print('Statistic: %.3f' % result.statistic)
    p = 0
    for i in range(len(result.critical_values)):
    	sl, cv = result.significance_level[i], result.critical_values[i]
    	if result.statistic < result.critical_values[i]:
    		print('%.3f: %.3f, Sample looks Gaussian (fail to reject H0)' % (sl, cv))
    	else:
    		print('%.3f: %.3f, Sample does not look Gaussian (reject H0)' % (sl, cv))


normality_test(residuals)


###############################################################################
"""Recommended model:
    SARIMA with seasonal 7 and ARMA(1,7) based on aic for flight 679 Economy
    the RMSE on validation set was 28.914
"""

###############################################################################
"Sarimax model with categorical values on full data"

# Variables
df.FlightID = df.FlightID.astype('object')
endog = df_train_697.loc[:, 'CabinBookings']
exog = sm.add_constant(df_train_697.loc[:, 'Cabin'])
nobs = endog.shape[0]

# Fit the model
mod = sm.tsa.statespace.SARIMAX(endog.values, exog=exog.values, order=(1,0,1))
fit_res = mod.fit(disp=False)
print(fit_res.summary())
