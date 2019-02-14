# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 12:57:21 2019

@author: Lenovo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\ML\\sabree travels\\dataset.csv")
df = df.drop("Forecasted Bookings",axis=1)
df.index = pd.to_datetime(df.ArrivalDate,format='%d-%m-%Y')

df_2014 = df.loc["2014"]
df_2015 = df.loc["2015"]

print("Unique flights IDs are..",df_2014.FlightID.unique())

df_2014_697 = df_2014[df_2014.FlightID==697]
df_2015_697 = df_2015[df_2015.FlightID==697]

ratio = df_2014_697.loc[:,"CabinBookings"]/df_2014_697.loc[:,"CabinCapacity"]
df_2014_697["BookingRatio"] = ratio.values

df_2014_697_bis = df_2014_697[df_2014_697.Cabin=="Business Cabin"]
df_2014_697_eco = df_2014_697[df_2014_697.Cabin=="Economy Cabin"]

"""Checking if the Booking Ratio has correlation with the Average Fare per Booking"""
print(" The correlation between booking ratio and average fare per booking..",df_2014_697_eco["BookingRatio"].corr(df_2014_697_eco["AverageFarePerBooking"]))

plt.scatter(df_2014_697_eco["BookingRatio"],df_2014_697_eco["AverageFarePerBooking"])
plt.show()

from scipy.stats import spearmanr
print("Spearman correlation..",spearmanr(df_2014_697_eco["BookingRatio"],df_2014_697_eco["AverageFarePerBooking"]))

from sklearn.preprocessing import PolynomialFeatures

X = df_2014_697_eco.loc[:,"BookingRatio"].values
y = df_2014_697_eco.loc[:,"AverageFarePerBooking"].values
poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X.reshape(X.shape[0],1))

import statsmodels.api as sm

lin_reg = sm.OLS( y,X_poly  ).fit()
print(lin_reg.pvalues)

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(poly.fit_transform(X.reshape(X.shape[0],1))), color = 'blue')
plt.title('poly')
plt.xlabel('Ratio')
plt.ylabel('Price')
plt.show()

"""  p values are significant .,"""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

time_series = df_2014_697_eco.loc[:,"AverageFarePerBooking"]
time_series.index = df_2014_697_eco.index
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

time_series.plot()
plt.show()

"""Seems like a time series with high variance and mean reveting.. lets decompose"""

from statsmodels.tsa.seasonal import seasonal_decompose
decomposed = seasonal_decompose(time_series, model='additive',freq = 30)
decomposed.plot()
plt.show()

"""I can observe some seasonal trends on considering 30 days of frequency"""

""" check using ACF, PACF plot"""
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARMA
from statsmodels.graphics.tsaplots import plot_pacf

#plotting the ACF
plot_acf(time_series,lags =20, alpha = 0.05)
"""up to 10 lags are being significant"""

plot_pacf(time_series,lags =20, alpha = 0.05)
"""lag 2, 4, 8 ,10,13  are significant"""

# Run the ADF test on the price series and print out the results
'''Augmented dickey fuller test
Null hypothesis : There is a unit root in the time series sample
Alternative Hypothesis : The series doesn't have a unit root., The series is stationary.'''

results = adfuller(time_series)
print("The p-value of the  adfuller is {}".format(results[1]))

if results[1] <= 0.05:
    print("Reject Null hypothesis, The series in statioary")
else:
    print("Do no Reject Null, The series is not stationary")

"""Using the first difference"""
time_series_diff = time_series.diff().dropna()

plot_acf(time_series_diff,lags =50, alpha = 0.05)
"""up to 10 lags are being significant"""

plot_pacf(time_series_diff,lags =50, alpha = 0.05)
"""lag 2, 4, 8 ,10,13  are significant"""


###############################################################################
"""Deciding a model"""

p =1
d =1
q =1
order = (p,d,q)
from statsmodels.tsa.arima_model import ARIMA
model_111 = ARIMA(time_series,order)
results = model_111.fit()

print("The model parameters of the AR and MA components are ...", results.params)
print("\n The p value of the parameters are ..", results.pvalues)

df_2015_697 = df_2015[df_2015.FlightID==697]
df_2015_697_eco = df_2015_697[df_2015_697.Cabin =="Economy Cabin"]
actual_values = df_2015_697_eco.loc[:,'AverageFarePerBooking'].values.reshape(df_2015_697_eco.shape[0],1)

forecasted_values = results.forecast(132)[0]

plt.plot(actual_values)
plt.plot(forecasted_values)
plt.show()
################################################################################
""" Trying out better models """

#aic_array = []
aic_matrix = np.zeros((15,15))
bic_matrix = np.zeros((15,15))

for p in range(0,15):
    for q in range(0,15):
        order = (p,1,q)
        try:
            model = ARIMA(time_series,order).fit()
            #aic_array.append(((p,1,q),round(model.aic)))
            aic_matrix[p,q] = round(model.aic)
            bic_matrix[p,q] = round(model.bic)
            print('ARIMA%s AIC=%.3f' % (order,round(model.aic)))
        except:
            continue

import seaborn as sns
sns.heatmap(aic_matrix, annot= True, fmt ='.4g')

################################################################################
"Final ARIMA Model"

"""Deciding a model"""

p =7
d =1
q =7
order = (p,d,q)
from statsmodels.tsa.arima_model import ARIMA
final_model = ARIMA(time_series,order).fit()

print("The model parameters of the AR and MA components are ...", final_model.params)
print("\n The p value of the parameters are ..", final_model.pvalues)

df_2015_697 = df_2015[df_2015.FlightID==697]
df_2015_697_eco = df_2015_697[df_2015_697.Cabin =="Economy Cabin"]
actual_values = df_2015_697_eco.loc[:,'AverageFarePerBooking'].values.reshape(df_2015_697_eco.shape[0],1)

forecasted_values = final_model.forecast(132)[0]

plt.plot(actual_values)
plt.plot(forecasted_values)
plt.show()

###############################################################################
""" Rolling Forecasts """
history = list(df_2014_697_eco.loc[:,'AverageFarePerBooking'].values)
test = list(df_2015_697_eco.loc[:,'AverageFarePerBooking'].values)
predictions = list()
for i in range(df_2015_697_eco.shape[0]):
	# predict
	yhat = history[-1]
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
    
plt.plot(test)
plt.plot(predictions)
plt.show()

# report performance
mse = mean_squared_error(test, predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)