import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.holtwinters as ets
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from scipy.stats import chi2
import seaborn as sns
from sklearn.metrics import mean_squared_error
from scipy.signal import dlsim

# non-seasonal differencing
def dif(dataset,interval=1):
    diff=[]
    for i in range(interval,len(dataset)):
        value=dataset[i]-dataset[i-interval]
        diff.append(value)
    return diff

# seasonal differencing
def dif_s(dataset,s):
    ''':param: s is the period'''
    interval = s
    diff=[]
    for i in range(interval,len(dataset),interval):
        value=dataset[i]-dataset[i-interval]
        diff.append(value)
    return diff

# Rolling mean & Rolling variance for sales, adbudget & GDP
def cal_rolling_mean_var(dat):
    rolldf = pd.DataFrame()
    m=[]
    v=[]
    for i in range(len(dat)):
        m.append(dat[0:i+1].mean())
        v.append(dat[0:i+1].var())
    rolldf['mean'] = m
    rolldf['var'] = v
    rolldf.fillna(0, inplace=True)
    #====
    fig = plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot('mean', data=rolldf, label="Mean", color='red')
    plt.title("Rolling Mean & Variance")
    plt.xticks([])
    plt.grid()
    plt.ylabel('mean')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot('var', data=rolldf, label="variance", color='blue')
    plt.grid()
    plt.ylabel('variance')
    plt.legend()
    plt.show()

# ADF-TEST
from statsmodels.tsa.stattools import adfuller
def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" %result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

# KPSS-TEST
from statsmodels.tsa.stattools import kpss
def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','LagsUsed'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
        print (kpss_output)


# Correlation function
def correlation_coefficient_cal(x, y):
    """ This returns the correlation of x and y values"""
    xm=np.mean(x)
    ym=np.mean(y)
    top=0; bot1=0; bot2=0
    for i in range(len(x)):
        top += (x[i]-xm)*(y[i]-ym)
        bot1 += (x[i]-xm)**2
        bot2 +=(y[i]-ym)**2
    bot1=np.sqrt(bot1)
    bot2 = np.sqrt(bot2)
    return top/(bot1*bot2)

#Auto-Correlation Function (ACF)
def ACF(array,lag,label):
    """
    This returns the Autocorrelation of a time series data.
    Print 'ac' to get autocorrelation coefficients

    Parameters
    ----------
    :param array :  Array of Time Series Data
    :param lag : Number of Lags
    :param label : Title
    """
    global ac
    mu= np.mean(array)
    var=0;Rt=0;ac=[]
    var= sum(map(lambda x:(x-mu)**2,array))
    for j in range(lag):
        for i in range((j),len(array)):
            Rt += (array[i]-mu)*(array[i-j]-mu)
        ac.append(round(Rt/var,4))
        Rt=0
    ry= ac[::-1][:-1] + ac
    plt.stem(np.linspace(-((len(ac))-1), len(ac)-1, (len(ac)*2-1), dtype=int), ry, markerfmt='o')
    plt.axhspan(-(1.96/np.sqrt(len(array))), (1.96/np.sqrt(len(array))), alpha=0.2, color='blue')
    plt.title(f'Autocorrelation Function of {label}')
    plt.ylabel('Magnitude'); plt.xlabel('Lags')
    plt.show()

# ACF_ RAW ONE SIDE VALUE
def acf_raw(array,lag):
    mu= np.mean(array)
    var=0;Rt=0;ac=[]
    var= sum(map(lambda x:(x-mu)**2,array))
    for j in range(lag):
        for i in range((j),len(array)):
            Rt += (array[i]-mu)*(array[i-j]-mu)
        ac.append(round(Rt/var,4))
        Rt=0
    ac
    return np.array(ac)

# Average Forecast
def average_forecast(array_train, array_test):
    """
    This function returns in order TRAIN FORECAST, TEST FORECAST,
    TRAIN_MSE, TEST_MSE using Average method
    :param: array_train- Numpy array for training dataset
    :param: array_test- Numpy array for testing dataset
    """
    global er1, er2
    array_train= np.array(array_train)
    array_test= np.array(array_test)
    l= len(array_train); y_forecast=[]
    for T in range(1,l+1):
        y_forecast.append((np.sum(array_train[:T]))/T)
    last_1step_forecast=y_forecast[-1]
    error= array_train[1:] - y_forecast[:-1]
    er1=error
    error= error**2
    mse= round(np.sum(error)/len(error),2)
    mse= mean_squared_error(array_train[1:],y_forecast[:-1])
    #==================================
    test_fcst=[]
    for hstep in range(len(array_test)):
        test_fcst.append(last_1step_forecast)
    test_fcst= np.array(test_fcst)
    er2=(array_test - test_fcst)
    error_f=(array_test - test_fcst)**2
    mse_f= round(np.sum(error_f)/len(error_f),2)
    mse_f= mean_squared_error(array_test,test_fcst)
    # print(f'MSE_Train: {mse}')
    # print(f'MSE_Test: {mse_f}')
    return np.round(np.array(y_forecast[:-1]),4), np.round(test_fcst,4), mse, mse_f,er1,er2

# Naive Forecast Method
def naive_forecast(array_train, array_test):
    """
    This function returns in order TRAIN FORECAST, TEST FORECAST,
    TRAIN_MSE, TEST_MSE using Naive method
    :param array_train- Numpy array for training dataset
    :param: array_test- Numpy array for testing dataset
    """
    global er1, er2
    array_train= np.array(array_train)
    array_test= np.array(array_test)
    train_f= []
    test_f=[]
    for T in range(len(array_train)):
        train_f.append(array_train[T])
    last_train_f=train_f[-1]
    train_f= np.array(train_f[:-1])
    er1=(array_train[1:]-train_f)
    error= (array_train[1:]-train_f)**2
    mse_train= np.sum(error)/len(error)
    mse_train= mean_squared_error(array_train[1:],train_f)
    #==================================
    for hstep in range(len(array_test)):
        test_f.append(last_train_f)
    test_f= np.array(test_f)
    er2=(array_test-test_f)
    error_2= (array_test-test_f)**2
    mse_test= np.sum(error_2)/len(error_2)
    mse_test= mean_squared_error(array_test,test_f)
    return np.round(train_f, 4), np.round(test_f, 4), mse_train, mse_test,er1, er2

# Drift Method
def drift_forecast(array_train, array_test):
    """
    This function returns in order TRAIN FORECAST, TEST FORECAST,
    TRAIN_MSE, TEST_MSE using Naive method
    :param array_train- Numpy array for training dataset
    :param: array_test- Numpy array for testing dataset
    """
    global er1, er2
    array_train = np.array(array_train)
    array_test = np.array(array_test)
    train_f = []
    test_f = []
    h=1
    for T in range(len(array_train)):
        y_hat= array_train[T]+ h*((array_train[T]-array_train[0])/(T))
        train_f.append(y_hat)
    # last_f= train_f[-1]
    train_f=np.array(train_f[1:-1])
    er1= (array_train[2:]-train_f)
    error1= (array_train[2:]-train_f)**2
    mse_train= np.sum(error1)/len(error1)
    mse_train=mean_squared_error(array_train[2:],train_f)
    #==================================
    # test_f.append(last_f)
    for T in range(len(array_test)):
        y_hat = array_train[len(array_train)-1] + (h*((array_train[len(array_train)-1] - array_train[0])) / (len(array_train) - 1))
        test_f.append(y_hat)
        h += 1
    er2= (array_test-test_f)
    error2= (array_test-test_f)**2
    mse_test= np.sum(error2)/len(error2)
    mse_test= mean_squared_error(array_test,test_f)
    return np.round(train_f, 4), np.round(test_f, 4), mse_train, mse_test,er1, er2


# Simple Exponential Smoothing
def SES(array_train, array_test,alpha, initial_condition):
    """
    This function returns in order TRAIN FORECAST, TEST FORECAST,
    TRAIN_MSE, TEST_MSE using Naive method
    :param array_train- Numpy array for training dataset
    :param: array_test- Numpy array for testing dataset
    :param: alpha- smoothing value
    :param: initial_condition- start-point for SES
    """
    global er1, er2
    array_train = np.array(array_train)
    array_test = np.array(array_test)
    train_f = []
    test_f = []
    yt= initial_condition
    train_f.append(yt)
    for t in range(len(array_train)):
        yt= alpha * array_train[t] + (1-alpha)*yt
        train_f.append(yt)
    last_f= train_f[-1]
    train_f= np.array(train_f[:-1])
    train_f= train_f[1:]
    er1= (array_train[1:] - train_f)
    error1= (array_train[1:] - train_f)**2
    mse_train= np.sum(error1)/len(error1)
    mse_train= mean_squared_error(array_train[1:],train_f)
    #====================================
    for i in range(len(array_test)):
        test_f.append(last_f)
    test_f = np.array(test_f)
    er2= (array_test - test_f)
    error2 = (array_test - test_f) ** 2
    mse_test = np.sum(error2) / len(error2)
    mse_test= mean_squared_error(array_test,test_f)
    return np.round(train_f, 4), np.round(test_f, 4), mse_train, mse_test,er1, er2


# Holt's Linear Trend
def holt_trend(train, test, trend,damped_trend, seasonal):
    """
    returns fitted, forecast, mse_pred, mse_forecast, var_pred, var_forecast
    :param train/test are train and test set as pandas series
    :param trend 'mul' or 'add'
    :param seasonal 'mul' or None or 'add'
    :param damped_trend True or False
    """
    global er1, er2, forecast_df
    model=ets.ExponentialSmoothing(train, trend=trend, damped_trend=damped_trend, seasonal=seasonal).fit()
    fitted= model.fittedvalues
    forecast= model.forecast(steps=len(test))
    forecast_df= pd.DataFrame(forecast).set_index(test.index)
    mse_p=np.mean(np.subtract(train,fitted)**2)
    mse_f= np.mean(np.subtract(test,forecast_df.iloc[:,0].values)**2)
    var_p= np.var(np.subtract(train,fitted))
    var_f = np.var(np.subtract(test, forecast_df.iloc[:,0].values))
    er1= np.subtract(train,fitted)
    er2 = np.subtract(test, forecast_df.iloc[:,0].values)
    return fitted, forecast, mse_p, mse_f, var_p, var_f,er1, er2

# Holt's Winter Seasonal Trend
def holt_trend_s(train, test, trend,damped_trend, seasonal, seasonal_periods):
    """
    returns fitted, forecast, mse_pred, mse_forecast, var_pred, var_forecast
    :param train/test are train and test set as numpy array
    :param trend 'mul' or 'add'
    :param seasonal 'mul' or 'add'
    :param seasonal_periods integer
    :param damped_trend True or False
    """
    global er1, er2
    model=ets.ExponentialSmoothing(train, trend=trend, damped_trend=damped_trend, seasonal=seasonal, seasonal_periods=seasonal_periods).fit()
    fitted= model.fittedvalues
    forecast= model.forecast(steps=len(test))
    forecast_df= pd.DataFrame(forecast).set_index(test.index)
    mse_p=np.mean(np.subtract(train,fitted)**2)
    mse_p= mean_squared_error(train,fitted)
    mse_f= np.mean(np.subtract(test,forecast_df.iloc[:,0].values)**2)
    mse_f= mean_squared_error(test,forecast_df.iloc[:,0].values)
    var_p= np.var(np.subtract(train,fitted))
    var_f = np.var(np.subtract(test, forecast_df.iloc[:,0].values))
    er1= np.subtract(train,fitted)
    er2 = np.subtract(test, forecast_df.iloc[:,0].values)
    return fitted, forecast, mse_p, mse_f, var_p, var_f,er1,er2 #remove er1,er2 to run assignment well

def graphit(train, test, test_f, method_name):
    arr= list(range(1,len(train)+len(test)+1))
    arr2= list(train) + list(test) + list(test_f)
    plt.plot(arr[0:len(train)],arr2[:len(train)],color='red',label='Training Data')
    plt.plot(arr[len(train):],arr2[len(train):len(train)+len(test)],color='blue',label='Testing Data')
    plt.plot(arr[len(train):], arr2[len(train)+len(test):],color='green',label='h-step forecast')
    plt.title(method_name)
    plt.legend(loc="upper left")
    plt.xlabel("Time")
    plt.ylabel("Time Series")
    plt.show()

# Box-Pierce Q-Value
def qvalue(acf, lag, length_of_observation):
    q= 0
    for i in range(1,lag):
        q += acf[i]**2
    q = q*length_of_observation
    # or q= length_of_observation*(np.sum(acf**2))
    return q

def ACF_PACF(y,lags):
    # ACF & PACF plots with AC generated from y
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    plt.rcParams['font.size']=13
    plt.figure(figsize=(12,10))
    plt.subplot(211)
    plt.title('ACF/PACF of RAW DATA')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    plt.show()

# def ACF_PACF(y,lags):
#     acf = sm.tsa.stattools.acf(y, nlags=lags)
#     pacf = sm.tsa.stattools.pacf(y, nlags=lags)
#     fig = plt.figure()
#     plt.subplot(211)
#     plt.title('ACF/PACF of the raw data')
#     plot_acf(y, ax=plt.gca(), lags=lags)
#     plt.subplot(212)
#     plot_pacf(y, ax=plt.gca(), lags=lags)
#     fig.tight_layout(pad=3)
#     plt.show()


def datetime_transformer(df, datetime_vars):
    """
    The datetime transformer

    Parameters
    ----------
    df : the dataframe
    datetime_vars : the datetime variables

    Returns
    ----------
    The dataframe where datetime_vars are transformed into the following 6 datetime types:
    year, month, day, hour, minute and second
    """

    # The dictionary with key as datetime type and value as datetime type operator
    dict_ = {'year': lambda x: x.dt.year,
             'month': lambda x: x.dt.month,
             'day': lambda x: x.dt.day,
             'hour': lambda x: x.dt.hour,
             'minute': lambda x: x.dt.minute,
             'second': lambda x: x.dt.second}

    # Make a copy of df
    df_datetime = df.copy(deep=True)

    # For each variable in datetime_vars
    for var in datetime_vars:
        # Cast the variable to datetime
        df_datetime[var] = pd.to_datetime(df_datetime[var])

        # For each item (datetime_type and datetime_type_operator) in dict_
        for datetime_type, datetime_type_operator in dict_.items():
            # Add a new variable to df_datetime where:
            # the variable's name is var + '_' + datetime_type
            # the variable's values are the ones obtained by datetime_type_operator
            df_datetime[var + '_' + datetime_type] = datetime_type_operator(df_datetime[var])

    # Remove datetime_vars from df_datetime
    df_datetime = df_datetime.drop(columns=datetime_vars)

    return df_datetime

# Ljung-Box test Q-value
# T= total num of samples h=lags rk=ACF of errors
def Q_ljbox(T,h,rk):
  """
  :param: T: total number of sample
  :param: h: lags
  :param: rk: ACF of Error
  """
  summ=[]
  for i in range(1,h):
    a= (1/(T-i))*(rk[i])**2
    summ.append(a)
  Q_star= T*(T+2)* np.sum(summ)
  return Q_star

# Ljung-Box Test with statsmodel
def Q_ljungbox(e, lags):
  q,pvalue= sm.stats.acorr_ljungbox(e,lags=[lags])
  print(f'Q: {q} \np-value: {pvalue}')

# Chi-Squared calculated value
def chi_square(alpha, DOF):
  chi_critical = chi2.ppf(1-alpha, DOF)
  return chi_critical

# Check for whiteness with Q and Chi-Squared
def Q_check(acf_error,lags,na,nb,n,alpha=0.01):
  Q= qvalue(acf_error,lags,n) # Q-value
  DOF= lags - na - nb # Degree of Freedom
  chi_critical= chi_square(alpha,DOF)
  print(f'Q-value: {Q}')
  print(f'Chi-Squared Critical: {chi_critical}')
  if Q< chi_critical:
    print('The residual is WHITE')
  else:
    print('The residual is NOT WHITE')

def gpac(na, nb,y,lags):
    '''

    :param na: Autoregressive order
    :param nb: Moving Average order
    :return: Generalized Partial Autocorrelation (GPAC) table and graph(GPAC, ACF, PACF)
    '''
    # ===========================================
        # ACF & PACF plots with AC generated from y
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    ry= acf
    plt.rcParams['font.size'] = 13
    plt.figure(figsize=(12, 10))
    plt.subplot(211)
    plt.title('ACF/PACF of RAW DATA')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    plt.show()
    #===========================================
    result = []
    k1 = [];k2 = [];k3 = [];k4 = [];k5 = [];k6 = [];k7 = []
    k8 = [];k9 = [];k10 = []
    for k in range(1, nb + 1):
        for j in range(na):
            if k == 1:
                k1.append(ry[j + k] / ry[j])
            if k == 2:
                numerator2 = [[ry[j], ry[j + 1]], [ry[abs(j + k - 1)], ry[j + k]]]
                denominator2 = [[ry[j], ry[abs(j - k + 1)]], [ry[abs(j + k - 1)], ry[j]]]
                k2.append(np.linalg.det(numerator2) / np.linalg.det(denominator2))
            if k == 3:
                numerator3 = [[ry[j], ry[abs(j - 1)], ry[j + 1]],
                              [ry[j + 1], ry[j], ry[j + 2]],
                              [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[j + k]]]
                denominator3 = [[ry[j], ry[abs(j - 1)], ry[abs(j - k + 1)]], [ry[j + 1], ry[j], ry[abs(j - k + 2)]],
                                [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[j]]]
                k3.append(np.linalg.det(numerator3) / np.linalg.det(denominator3))
            if k == 4:
                numerator4 = [[ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[j + 1]],
                              [ry[j + 1], ry[j], ry[abs(j - 1)], ry[j + 2]],
                              [ry[j + 2], ry[j + 1], ry[j], ry[j + 3]],
                              [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[j + k]]]

                denominator4 = [[ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - k + 1)]],
                                [ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - k + 2)]],
                                [ry[j + 2], ry[j + 1], ry[j], ry[abs(j - k + 3)]],
                                [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[j]]]

                k4.append(np.linalg.det(numerator4) / np.linalg.det(denominator4))
            if k == 5:
                numerator5 = [[ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[j + 1]],
                              [ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[j + 2]],
                              [ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[j + 3]],
                              [ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j + 4]],
                              [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)],
                               ry[j + k]]]

                denominator5 = [[ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - k + 1)]],
                                [ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - k + 2)]],
                                [ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - k + 3)]],
                                [ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - k + 4)]],
                                [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)], ry[j]]]
                k5.append(np.linalg.det(numerator5) / np.linalg.det(denominator5))
            if k == 6:
                numerator6 = [[ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[j + 1]],
                              [ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[j + 2]],
                              [ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[j + 3]],
                              [ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[j + 4]],
                              [ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j + 5]],
                              [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)],
                               ry[abs(j + k - 5)], ry[j + k]]]
                denominator6 = [
                    [ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[abs(j - k + 1)]],
                    [ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - k + 2)]],
                    [ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - k + 3)]],
                    [ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - k + 4)]],
                    [ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - k + 5)]],
                    [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)], ry[abs(j + k - 5)],
                     ry[j]]]
                k6.append(np.linalg.det(numerator6) / np.linalg.det(denominator6))

            if k == 7:
                numerator7 = [
                    [ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[abs(j - 5)], ry[j + 1]],
                    [ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[j + 2]],
                    [ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[j + 3]],
                    [ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[j + 4]],
                    [ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[j + 5]],
                    [ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j + 6]],
                    [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)], ry[abs(j + k - 5)],
                     ry[abs(j + k - 6)],
                     ry[j + k]]]
                denominator7 = [[ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[abs(j - 5)],
                                 ry[abs(j - k + 1)]],
                                [ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)],
                                 ry[abs(j - k + 2)]],
                                [ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)],
                                 ry[abs(j - k + 3)]],
                                [ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)],
                                 ry[abs(j - k + 4)]],
                                [ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - k + 5)]],
                                [ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - k + 6)]],
                                [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)],
                                 ry[abs(j + k - 5)],
                                 ry[abs(j + k - 6)], ry[j]]]

                k7.append(np.linalg.det(numerator7) / np.linalg.det(denominator7))

            if k == 8:
                numerator8 = [[ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[abs(j - 5)],
                               ry[abs(j - 6)], ry[j + 1]],
                              [ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)],
                               ry[abs(j - 5)], ry[j + 2]],
                              [ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)],
                               ry[abs(j - 4)], ry[j + 3]],
                              [ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)],
                               ry[j + 4]],
                              [ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)],
                               ry[j + 5]],
                              [ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[j + 6]],
                              [ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[j + 7]],
                              [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)],
                               ry[abs(j + k - 5)], ry[abs(j + k - 6)],
                               ry[abs(j + k - 7)], ry[j + k]]]

                denominator8 = [[ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[abs(j - 5)],
                                 ry[abs(j - 6)], ry[abs(j - k + 1)]],
                                [ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)],
                                 ry[abs(j - 5)], ry[abs(j - k + 2)]],
                                [ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)],
                                 ry[abs(j - 4)], ry[abs(j - k + 3)]],
                                [ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)],
                                 ry[abs(j - k + 4)]],
                                [ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)],
                                 ry[abs(j - k + 5)]],
                                [ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)],
                                 ry[abs(j - k + 6)]],
                                [ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],
                                 ry[abs(j - k + 7)]],
                                [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)],
                                 ry[abs(j + k - 5)],
                                 ry[abs(j + k - 6)], ry[abs(j + k - 7)], ry[j]]]

                k8.append(np.linalg.det(numerator8) / np.linalg.det(denominator8))

            if k == 9:
                numerator9 = [[ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[abs(j - 5)],
                               ry[abs(j - 6)], ry[abs(j - 7)], ry[j + 1]],
                              [ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)],
                               ry[abs(j - 5)], ry[abs(j - 6)], ry[j + 2]],
                              [ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)],
                               ry[abs(j - 4)], ry[abs(j - 5)], ry[j + 3]],
                              [ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)],
                               ry[abs(j - 4)],
                               ry[j + 4]],
                              [ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)],
                               ry[abs(j - 3)],
                               ry[j + 5]],
                              [ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)],
                               ry[abs(j - 2)], ry[j + 6]],
                              [ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)],
                               ry[j + 7]],
                              [ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],
                               ry[j + 8]],
                              [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)],
                               ry[abs(j + k - 5)], ry[abs(j + k - 6)],
                               ry[abs(j + k - 7)], ry[abs(j + k - 8)], ry[j + k]]]

                denominator9 = [[ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[abs(j - 5)],
                                 ry[abs(j - 6)], ry[abs(j - 7)], ry[abs(j - k + 1)]],
                                [ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)],
                                 ry[abs(j - 5)], ry[abs(j - 6)], ry[abs(j - k + 2)]],
                                [ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)],
                                 ry[abs(j - 4)], ry[abs(j - 5)], ry[abs(j - k + 3)]],
                                [ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)],
                                 ry[abs(j - 4)],
                                 ry[abs(j - k + 4)]],
                                [ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)],
                                 ry[abs(j - 3)],
                                 ry[abs(j - k + 5)]],
                                [ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)],
                                 ry[abs(j - 2)], ry[abs(j - k + 6)]],
                                [ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],
                                 ry[abs(j - 1)],
                                 ry[abs(j - k + 7)]],
                                [ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],
                                 ry[abs(j - k + 8)]],
                                [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)],
                                 ry[abs(j + k - 5)],
                                 ry[abs(j + k - 6)],
                                 ry[abs(j + k - 7)], ry[abs(j + k - 8)], ry[j]]]

                k9.append(np.linalg.det(numerator9) / np.linalg.det(denominator9))

            if k == 10:
                numerator10 = [[ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[abs(j - 5)],
                                ry[abs(j - 6)], ry[abs(j - 7)], ry[abs(j - 8)], ry[j + 1]],
                               [ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)],
                                ry[abs(j - 5)], ry[abs(j - 6)], ry[abs(j - 7)], ry[j + 2]],
                               [ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)],
                                ry[abs(j - 4)], ry[abs(j - 5)], ry[abs(j - 6)], ry[j + 3]],
                               [ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)],
                                ry[abs(j - 4)], ry[abs(j - 5)],
                                ry[j + 4]],
                               [ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)],
                                ry[abs(j - 3)], ry[abs(j - 4)],
                                ry[j + 5]],
                               [ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)],
                                ry[abs(j - 2)], ry[abs(j - 3)], ry[j + 6]],
                               [ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)],
                                ry[abs(j - 2)],
                                ry[j + 7]],
                               [ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],
                                ry[abs(j - 1)],
                                ry[j + 8]],
                               [ry[j + 8], ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1],
                                ry[j],
                                ry[j + 9]],
                               [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)],
                                ry[abs(j + k - 5)],
                                ry[abs(j + k - 6)],
                                ry[abs(j + k - 7)], ry[abs(j + k - 8)], ry[abs(j + k - 9)], ry[j + k]]]

                denominator10 = [[ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)], ry[abs(j - 5)],
                                  ry[abs(j - 6)], ry[abs(j - 7)], ry[abs(j - 8)], ry[abs(j - k + 1)]],
                                 [ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - 4)],
                                  ry[abs(j - 5)], ry[abs(j - 6)], ry[abs(j - 7)], ry[abs(j - k + 2)]],
                                 [ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)], ry[abs(j - 3)],
                                  ry[abs(j - 4)], ry[abs(j - 5)], ry[abs(j - 6)], ry[abs(j - k + 3)]],
                                 [ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)],
                                  ry[abs(j - 3)],
                                  ry[abs(j - 4)], ry[abs(j - 5)],
                                  ry[abs(j - k + 4)]],
                                 [ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)], ry[abs(j - 2)],
                                  ry[abs(j - 3)], ry[abs(j - 4)],
                                  ry[abs(j - k + 5)]],
                                 [ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j], ry[abs(j - 1)],
                                  ry[abs(j - 2)], ry[abs(j - 3)], ry[abs(j - k + 6)]],
                                 [ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],
                                  ry[abs(j - 1)], ry[abs(j - 2)],
                                  ry[abs(j - k + 7)]],
                                 [ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2], ry[j + 1], ry[j],
                                  ry[abs(j - 1)],
                                  ry[abs(j - k + 8)]],
                                 [ry[j + 8], ry[j + 7], ry[j + 6], ry[j + 5], ry[j + 4], ry[j + 3], ry[j + 2],
                                  ry[j + 1], ry[j],
                                  ry[abs(j - k + 9)]],
                                 [ry[abs(j + k - 1)], ry[abs(j + k - 2)], ry[abs(j + k - 3)], ry[abs(j + k - 4)],
                                  ry[abs(j + k - 5)],
                                 ry[abs(j + k - 6)],
                                  ry[abs(j + k - 7)], ry[abs(j + k - 8)], ry[abs(j + k - 9)], ry[j]]]
                k10.append(np.linalg.det(numerator10) / np.linalg.det(denominator10))
    if len(k10) > 0:
        result.append(k10)
    if len(k9) > 0:
        result.append(k9)
    if len(k8) > 0:
        result.append(k8)
    if len(k7) > 0:
        result.append(k7)
    if len(k6) > 0:
        result.append(k6)
    if len(k5) > 0:
        result.append(k5)
    if len(k4) > 0:
        result.append(k4)
    if len(k3) > 0:
        result.append(k3)
    if len(k2) > 0:
        result.append(k2)
    if len(k1) > 0:
        result.append(k1)
    result = pd.DataFrame(np.round(result[::-1], 3), index=list(range(1, nb + 1)), columns=list(range(na))).T
    plt.rcParams['figure.figsize'] = [10, 9]
    plt.rcParams['font.size'] = 11
    sns.heatmap(result, annot=True, xticklabels=list(range(1, nb + 1)), yticklabels=list(range(na)))
    plt.title('Generalized Partial Autocorrelation (GPAC)')
    plt.gcf().set_size_inches(15, 8)
    plt.show()
    return result

# generate data
# WN(0,1)
def data_gen():
    np.random.seed(42)
    n = int(input('Enter the number of samples: '))
    mean = float(input('Enter the mean: '))
    var = float(input('Enter the variance: '))
    var = np.sqrt(var)
    na = int(input('Enter AR order: '))
    nb = int(input('Enter MA order: '))
    AR = list(eval(input('Input the AR coefficients seperated by comma (start with 1): ')))
    MA = list(eval(input('Input the MA coefficients  seperated by comma (start with 1): ')))
    AR = np.array(AR)
    MA = np.array(MA)
    ARMA_process = sm.tsa.ArmaProcess(AR, MA)
    mean_y = (mean * (1 + np.sum(MA[1:]))) / (1 + np.sum(AR[1:]))  # Theoretical mean
    y = ARMA_process.generate_sample(n, scale=np.sqrt(var)) + mean_y  # plus theoretical mean (if mean is not 0 and var is not 1)
    return y

#==============================================
# Developing Levenberg-Marquardt (LM) Algorithm
def gen_e(y,num, den):
    '''
    :param y: time series values
    :param num: list of numerator values (MA params)
    :param den: list of denominator values (AR params)
    :return: errors
    '''
    np.random.seed(42)
    y=np.array(y)
    sys= (den, num,1)
    _,e= dlsim(sys,y)
    e= e.ravel()
    # e[(e == -(np.inf)) | (e == np.inf) | (e == np.nan)] = 0
    e[np.isnan(e)] = np.zeros(1)[0]
    e[np.isinf(e)] = np.zeros(1)[0]
    return np.array(e)

def LMA_params(teta, e, delta, na, nb):
    '''
    :param teta: list of coefficients
    :param e: list of error
    :param delta: update parameter
    :param na: int of AR order
    :param nb: int of MA order
    :return: X, A, g
    '''
    e=e.reshape(-1,1)
    n= na + nb
    N= len(e)
    x=[]
    # tetas = []
    teta= teta.ravel()
    p1=teta[:na] #AR params
    p1=np.append(p1,[0]*(np.maximum(na,nb)-na))
    p2=teta[na:] #MA params
    p2=np.append(p2,[0]*(np.maximum(na,nb)-nb))
    e_orig = gen_e(y, den=np.r_[1,p1], num=np.r_[1,p2])  # regular teta errors
    e_orig= e_orig.reshape(-1,1)
    for i in range(n):
        a= teta.copy()
        a[i]= a[i]+ delta
        # tetas.append(a)
        p11 = a[:na]  # AR params
        p11 = np.r_[p11, [0] * (np.maximum(na, nb) - na)]
        p22 = a[na:]  # MA params
        p22 = np.r_[p22, [0] * (np.maximum(na, nb) - nb)]
        e_new = gen_e(y, den=np.r_[1, p11], num=np.r_[1, p22])  # updated tetas errors
        e_new = e_new.reshape(-1, 1)
        sol = (e_orig - e_new) / delta
        sol= sol.ravel()
        sol = sol.tolist()
        x.append(sol)
    X=np.array(x)
    X= (X.T).reshape(-1,n)
    A= X.T @ X
    g= X.T @ e
    g[np.isnan(g)] = np.zeros(1)[0]
    g[np.isinf(g)] = np.zeros(1)[0]
    return X, A, g


def SSE(e):
    '''
    :param e: numpy array of error
    :return: Sum of Squared Error
    '''
    # e[(e == -(np.inf)) | (e == np.inf)| (e == np.nan)] = 0
    e[np.isnan(e)] = np.zeros(1)[0]
    e[np.isinf(e)] = np.zeros(1)[0]
    sse= e.T @ e
    return sse

def teta_change(A, mu, g,na, nb):
    n= na + nb
    I= np.eye(n,n)
    change_in_teta= np.linalg.inv(A + mu*I)
    change_in_teta= change_in_teta @ g
    return change_in_teta

def teta_new(teta_old, change_in_teta):
    new= teta_old + change_in_teta
    return new


def initial(y,na,nb,mu=0.01):
    global teta,X,A,g,sse_old,sse_new,ch_teta,ch_teta_norm,new_teta,e
    # =========================
    maxx= np.maximum(na,nb)
    n = na + nb
    AR, MA = np.r_[1, [0] * maxx], np.r_[1, [0] * maxx]  # make params zeros
    N = len(y)
    e = gen_e(y,num=MA,den=AR)
    e = e.reshape(-1, 1)
    delta = 0.000001
    teta = [0] * (na + nb)
    teta = np.array(teta,dtype='float').reshape(-1, 1)
    sse_old = SSE(e)
    X, A, g = LMA_params(teta, e, delta, na, nb)
    #sse_old = SSE(e)
    ch_teta = teta_change(A, mu, g, na, nb)
    new_teta = teta_new(teta, ch_teta)
    pp1=new_teta[:na].ravel() #AR params
    pp1= np.r_[pp1,[0]*(np.maximum(na,nb)-na)]
    pp2= new_teta[na:].ravel() #MA params
    pp2= np.r_[pp2,[0]*(np.maximum(na,nb)-nb)]
    AR = np.r_[1,pp1]
    MA = np.r_[1,pp2]
    e = gen_e(y, num=MA, den=AR)
    e= e.reshape(-1,1)
    sse_new = SSE(e)
    ch_teta_norm = norm(ch_teta)
    # ==========================

# teta= new_teta
def step1(teta,na,nb,N,e):
    global X,A,g,sse_old
    e=e.reshape(-1,1)
    # ==========================
    ppp1 = teta[:na].ravel()  # AR params
    ppp1 = np.r_[ppp1, [0] * (np.maximum(na, nb) - na)]
    ppp2 = teta[na:].ravel()  # MA params
    ppp2 = np.r_[ppp2, [0] * (np.maximum(na, nb) - nb)]
    AR= np.r_[1,ppp1]
    MA= np.r_[1,ppp2]
    e = gen_e(y, num=MA, den=AR)
    e= e.reshape(-1,1)
    n = na + nb
    delta = 0.0000001
    X, A, g = LMA_params(teta, e, delta, na, nb)
    sse_old = SSE(e)

def step2(A,g,teta,na,nb,mu=0.01):
    global ch_teta,new_teta,e,sse_new,ch_teta_norm,var_error,cov_teta
    n=na+nb
    ch_teta = teta_change(A, mu, g, na, nb)
    new_teta = teta_new(teta, ch_teta)

    pppp1 = new_teta[:na].ravel()  # AR params
    pppp1 = np.r_[pppp1, [0] * (np.maximum(na, nb) - na)]
    pppp2 = new_teta[na:].ravel()  # MA params
    pppp2 = np.r_[pppp2, [0] * (np.maximum(na, nb) - nb)]

    AR = np.r_[1,pppp1]
    MA = np.r_[1,pppp2]
    e = gen_e(y, num=MA, den=AR)
    e= e.reshape(-1,1)
    sse_new = SSE(e)
    #===========
    if np.isnan(sse_new):
        sse_new = 10 ** 10
    #===========
    # ch_teta[(ch_teta == -(np.inf)) | (ch_teta == np.inf) | (ch_teta == np.nan)] = 0.0
    # ch_teta[np.isnan(ch_teta)] = np.zeros(1)[0]
    # ch_teta[np.isinf(ch_teta)] = np.zeros(1)[0]
    # ch_teta_norm = norm(ch_teta)
    # var_error = sse_new / (N - n)
    # A[np.isnan(A)] = np.zeros(1)[0]
    # A[np.isinf(A)] = np.zeros(1)[0]
    # cov_teta = var_error * np.linalg.pinv(A)
    # ==========================


def levenberg_marquardt(y,na,nb,e,mu=0.01,max_iter=100,n_iter=0,act=False):
    global teta, teta_estimate, cov_teta
    n=na+nb
    if act==False:
        if n_iter < max_iter:
            if sse_new< sse_old:
                if ch_teta_norm< 0.001:
                    teta_estimate= new_teta
                    var_error = sse_new / (N - n)
                    cov_teta = var_error * np.linalg.pinv(A)
                    print('===Results Available===')
                    print(teta_estimate)
                    act = True
                else:
                    teta= new_teta
                    mu= mu/10
                    step1(teta, na, nb, N, e)
                    step2(A, g, teta, na, nb, mu)
            while sse_new >= sse_old:
                mu=mu*10
                #step2(A, g, teta, na, nb, mu=mu)
                step2(A, g, teta, na, nb, mu=mu)
                if mu>1e20: #1e20
                    # print(f' "max_mu"')
                    print(new_teta)
                    act=True
                    n_iter=max_iter
                    break
                # step2(A,g,teta,na,nb,mu=mu)
            n_iter += 1
            if n_iter> max_iter:
                # print(f'Error: "max_iteration" ')
                # print(new_teta)
                act = True

            teta= new_teta
            step1(teta,na,nb,N,e)
            step2(A,g,teta,na,nb,mu)
            levenberg_marquardt(y, na, nb, e=e, mu=mu, max_iter=50, n_iter=n_iter)
        else:
            pass
    else:
        pass

global y,N
def LM_algorithm(y,na,nb):
    '''
    :param y: numpy array of values
    :param na: AR order
    :param nb: MA order
    :return: estimated parameters and covariance(as global variable)
    '''
    with np.errstate(divide='ignore'):
        np.float64(1.0) / 0.0
    initial(y,na,nb)
    levenberg_marquardt(y,na,nb,e)
    return new_teta
#==========================================================