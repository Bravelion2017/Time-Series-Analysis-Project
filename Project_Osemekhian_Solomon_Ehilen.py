#-----------Importing Libraries--------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.holtwinters as ets
import seaborn as sns
import datetime
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import register_matplotlib_converters
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import STL
from scipy import signal
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from scipy.signal import dlsim
import statsmodels.api as sm
from scipy.linalg import norm
pd.set_option('display.max_columns', None)
import sys
sys.path.append(r'C:\Users\oseme\PycharmProjects\pythonProject2\mytools')
from mytools import *
import warnings
warnings.filterwarnings('ignore')
random_seed=42
target='traffic_volume'
#--------------------------------------------------------------
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



def levenberg_marquardt(y,na,nb,e,mu=0.01,max_iter=100,n_iter=0,act=False):
    global teta, teta_estimate, cov_teta
    n=na+nb
    sse_tracker.append(sse_new)
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
            var_error = sse_new / (N - n)
            cov_teta = var_error * np.linalg.pinv(A)
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


def LM_algorithm(y,na,nb):
    '''
    :param y: numpy array of values
    :param na: AR order
    :param nb: MA order
    :return: estimated parameters and covariance(as global variable)
    '''
    global sse_tracker
    sse_tracker=[]
    with np.errstate(divide='ignore'):
        np.float64(1.0) / 0.0
    initial(y,na,nb)
    sse_tracker.append(sse_old)
    levenberg_marquardt(y,na,nb,e)
    return new_teta
#==========================================================
def zero_pole(na,nb,teta):
    ar= np.r_[1,teta[:na]]
    ma = np.r_[1, teta[na:]]
    print(f'Zero: {np.roots(ma)}')
    print(f'Pole: {np.roots(ar)}')

def print_coef(na,nb,model_obj):
  for i in range(na):
    print(f'The AR Coefficient a{i+1}: {np.negative(model_obj.params[i])}')
  for i in range(nb):
    print(f'The MA Coefficient b{i+1}: {model_obj.params[na+i]}')
#==========================================================
#---------Importing Dataset------------------------------------
df= pd.read_csv('Metro_Interstate_Traffic_Volume.csv')
df_copy= df.copy(deep=True)
datetime= pd.date_range(start='2012-10-02 09:00:00',
                        periods=len(df_copy), freq='1H') #set accurate date_range
df_copy['datetime']=datetime
#df_copy['date_time']=pd.DatetimeIndex(df_copy['date_time']) # change to type- datetime
# pd.DatetimeIndex(df_copy['date_time']).year

#---------------Missing Values & transformation---------------------------------
print(f'Missing Values: {df_copy.isna().sum().any()}')
df_copy2=datetime_transformer(df_copy,['date_time'])
df_copy2.drop(columns=['date_time_minute','date_time_second'],inplace=True)
df_copy_dum= pd.get_dummies(df_copy2,columns=['holiday','weather_main','weather_description'])

#---------------Dependent Variable time Plot-------------------
df_copy.plot(x='datetime',y='traffic_volume')
plt.title('Traffic Volume')
plt.xlabel('Time')
plt.ylabel('Volume of Traffic')
plt.xticks(rotation=45)
plt.show()

#---------------ACF/PACF Plot-------------------
ACF_PACF(df_copy[target],50)

#---------------Heatmap-------------------
cor= df_copy.corr(method='pearson')
sns.heatmap(cor, annot=True)
plt.title('Correlation Between All Features')
plt.show()

#----------------Splitting Dataset-----------------------------
df_train, df_test= train_test_split(df_copy2,train_size=0.8,shuffle=False)

#----------------Stationarity Check-----------------------------
# Initial check for stationarity on training and testing data set

# call rolling mean/variance
cal_rolling_mean_var(df_copy[target])
# ADF Test
ADF_Cal(df_copy[target])
# KPSS Test
kpss_test(df_copy[target])
ACF(df_copy[target],50,'Traffic Volume')


# Time Series Decomposition with
# STL (Seasonal and Trend Decomposition using Loess)
datetime= pd.date_range(start='2012-10-02 09:00:00',
                        periods=len(df_copy), freq='1H') #set accurate date_range
traffic_volume= pd.Series(np.array(df_copy[target]),
                          index=datetime)

STL= STL(traffic_volume)
result= STL.fit()
plt.figure(figsize=(25,20))
fig= result.plot()
# fig.suptitle('STL Decomposition for Traffic Volume')
plt.show()

# Seasonality, Trend & Residual
T= result.trend
S= result.seasonal
R= result.resid
# strength of seasonality and trend
F= np.maximum(0,1- np.var(np.array(R))/np.var(np.array(S)+np.array(R)))

print(f'The strength of seasonality for this data set is: {F:.4f}')
F2= np.maximum(0,1- np.var(np.array(R))/np.var(np.array(T)+np.array(R)))
print(f'The strength of trend for this data set is: {F2:.4f}')

# Adjusted seasonality
adj_seasonal= df_copy[target].values - S.values # additive decomposition

plt.plot(datetime,df_copy[target],label='Original')
plt.plot(datetime,adj_seasonal,label='Seasonality Adjusted')
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Seasonality Adjusted vs Original Dataset')
plt.legend()
plt.show()

# Adjusted trend
adj_trend= df_copy[target].values - T.values # additive decomposition

plt.plot(datetime,df_copy[target],label='Original')
plt.plot(datetime,adj_trend,label='Trend Adjusted')
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Trend Adjusted vs Original Dataset')
plt.legend()
plt.show()


# GPAC
gpac(10,10,y=df_copy['traffic_volume'],lags=40)

# Empty Dataframe
check = pd.DataFrame(columns=['Q','MSE_pred','MSE_f','var_pred_er','var_forecast_er'])

# Holt-Winter Method of forecasting
# This considers the trend and seasonality
fitted, forcast,mse_pred, mse_f, var_p,var_f,er1,er2= holt_trend_s(df_train['traffic_volume'],
                                                                   df_test['traffic_volume'],'add',True,'add',seasonal_periods=24) #TES #er1-residual error er2-forecast error
print(f'Train MSE: {mse_pred}')
print(f'Test MSE: {mse_f}')
# There is an overfitting problem with Holt-Winter

print(f'Variance: Prediction variance: {var_p} Forecast variance: {var_f}')
print(f'Ratio between variance of prediction & forecast: {var_p/var_f}')
# The ratio is very far from 1. This is a bad model for this dataset

# acf= acf_raw(er1,100)
# q= qvalue(acf, 100,len(df_train['traffic_volume'])+len(df_test['traffic_volume']))
print('Ljung-Box test:')
Q= sm.stats.acorr_ljungbox(er1, lags=[100],return_df=True)
print(Q)
# null hypothesis: The residual is white noise
# alternate hypothesis: The residual is not white noise
# The residual error is not white noise with p-value less than 0.05
hw_pred=fitted
hw_forecast= forcast
check.loc["Holt-Winters mtd"]= [Q.lb_stat.ravel()[0],mse_pred,mse_f, round(np.var(er1),2),round(np.var(er2),2)]


# Graphing forecast and ACF of errors
#=====
def plot_pred(train,fitted,label):
    plt.plot(train, label='Train')
    plt.plot(fitted, label='Predicted')
    plt.title(label)
    plt.xlabel('Hour')
    plt.ylabel('Traffic Volume')
    plt.legend()
    plt.show()
#====
graphit(df_train['traffic_volume'], df_test['traffic_volume'], forcast, "Forecast Holt-Winters Method")
plot_pred(df_train['traffic_volume'],fitted,"Prediction with Holt-Winters Method")
ACF(er1,50,'Holt-Winter Method on Traffic Volume')
# e=df_train['traffic_volume'][1:]-fitted[:-1]
# Residual ACF plot is not a white noise


#   Feature Selection w/Backward stepwise regression

# SVD and Condition Number without categorical variables
x= df_train[['temp','rain_1h','snow_1h','clouds_all']]
cols= x.columns
X= x.values
H= np.matmul(X.T,X)
#SVD
_,d,_=np.linalg.svd(H)
res=pd.DataFrame(d,index=cols, columns=['Singular Values'])
print(res)
print(f'Condition number for X is {np.linalg.cond(x)}')
# The least singular value 3.2 is closer to zero compared to the other values.
# This indicate presence of co-linear feature(s).
# The condition number indicates severe degree of co-linearity with condition number greater than 1000.

# OLS with statsmodel without cat variables
x= sm.add_constant(x) # added constant 1 vector
y= df_train['traffic_volume'].values
model= sm.OLS(y,x).fit()
print(model.summary())

# Remove feature const
model= sm.OLS(y,x.drop(columns=['const'])).fit()
print(model.summary())

# Remove feature snow_all with p>|t| 0.912
model= sm.OLS(y,x.drop(columns=['const','snow_1h'])).fit()
print(model.summary())

# Remove feature rain_1h with p>|t| 0.409
final_model= sm.OLS(y,x.drop(columns=['const','snow_1h','rain_1h'])).fit()
print(final_model.summary())
# The final model has the best F-statistics with p-value and a significant
# t-value for the two features left (temp & clouds_all)
xnew= df_train[['temp','clouds_all']]
print(f'Condition number after Feature Selection for X is {np.linalg.cond(xnew)}')

# Base Models

# Average Forecast
train_f, test_f, mse_train, mse_test,er1,er2= average_forecast(df_train['traffic_volume'],df_test['traffic_volume'])
print(f'Train MSE: {mse_train}')
print(f'Test MSE: {mse_test}')
print('Ljung-Box test:')
Q= sm.stats.acorr_ljungbox(er1, lags=[100],return_df=True)
print(Q)
check.loc["Average mtd"]= [Q.lb_stat.ravel()[0],mse_train,mse_test, round(np.var(er1),2),round(np.var(er2),2)]
graphit(df_train['traffic_volume'], df_test['traffic_volume'], test_f, "Average Forecast Method")
plot_pred(df_train['traffic_volume'].values,train_f.ravel(),"Prediction with Average Method")
ACF(er1,50,'Average Method on Traffic Volume')
avg_pred=train_f
avg_forecast= test_f


# Naive Forecast Method
train_f, test_f, mse_train, mse_test,er1,er2= naive_forecast(df_train['traffic_volume'],df_test['traffic_volume'])
print(f'Train MSE: {mse_train}')
print(f'Test MSE: {mse_test}')
print('Ljung-Box test:')
Q= sm.stats.acorr_ljungbox(er1, lags=[100],return_df=True)
print(Q)
check.loc["Naive mtd"]= [Q.lb_stat.ravel()[0],mse_train, mse_test, round(np.var(er1),2),round(np.var(er2),2)]
graphit(df_train['traffic_volume'], df_test['traffic_volume'], test_f, "Naive Forecast Method")
plot_pred(df_train['traffic_volume'].values,train_f.ravel(),"Prediction with Naive Method")
ACF(er1,50,'Naive Method on Traffic Volume')
naive_pred=train_f
naive_forecast= test_f

# Drift Forecast Method
train_f, test_f, mse_train, mse_test,er1,er2= drift_forecast(df_train['traffic_volume'],df_test['traffic_volume'])
print(f'Train MSE: {mse_train}')
print(f'Test MSE: {mse_test}')
print('Ljung-Box test:')
Q= sm.stats.acorr_ljungbox(er1, lags=[100],return_df=True)
print(Q)
check.loc["Drift mtd"]= [Q.lb_stat.ravel()[0],mse_train, mse_test, round(np.var(er1),2),round(np.var(er2),2)]
graphit(df_train['traffic_volume'], df_test['traffic_volume'], test_f, "Drift Forecast Method")
plot_pred(df_train['traffic_volume'].values,train_f.ravel(),"Prediction with Drift Method")
ACF(er1,50,'Drift Method on Traffic Volume')
drift_pred=train_f
drift_forecast= test_f

# Simple Exponential Smoothing
train_f, test_f, mse_train, mse_test,er1,er2= SES(df_train['traffic_volume'],df_test['traffic_volume'],alpha=0.5,
                                                  initial_condition=df_train['traffic_volume'][0])
print(f'Train MSE: {mse_train}')
print(f'Test MSE: {mse_test}')
print('Ljung-Box test:')
Q= sm.stats.acorr_ljungbox(er1, lags=[100],return_df=True)
print(Q)
check.loc["SES mtd"]= [Q.lb_stat.ravel()[0],mse_train, mse_test, round(np.var(er1),2),round(np.var(er2),2)]
graphit(df_train['traffic_volume'], df_test['traffic_volume'], test_f, "Simple Exponential Smoothing")
plot_pred(df_train['traffic_volume'].values,train_f.ravel(),"Simple Exponential Smoothing")
ACF(er1,50,'Simple Exponential Smoothing Method on \nTraffic Volume')
ses_pred=train_f
ses_forecast= test_f

# Multiple linear Regression Using Least Square
x= df_copy2[['temp','clouds_all']]
cols= x.columns
X= x.values
y= df_copy2['traffic_volume'].values
#check for collinearity
H= np.matmul(X.T,X)
#SVD
_,d,_=np.linalg.svd(H)
res=pd.DataFrame(d,index=cols, columns=['Singular Values'])
print(res)
print(f'Condition number for X is {np.linalg.cond(x)}')

#Splitting into train and test sets
xtrain,xtest,ytrain,ytest= train_test_split(x, y, test_size=0.2,shuffle=False)

# regression model
reg_model= sm.OLS(ytrain,xtrain).fit()
print(reg_model.summary())
# Regression equation
reg_fitted= reg_model.predict(xtrain)
reg_forecast= reg_model.predict(xtest)

residual= ytrain[1:] - reg_fitted[:-1]
forecast_error= ytest[2:] - reg_forecast[:-2]
ACF(residual,50,'Residual on Multiple Linear Regression')
print('Ljung-Box test:')
Q= sm.stats.acorr_ljungbox(residual, lags=[100],return_df=True)
print(Q)
mse_pred=mean_squared_error(ytrain,reg_fitted)
print(f'MSE of prediction: {mse_pred}')
from sklearn.metrics import r2_score
r2=r2_score(ytrain,reg_fitted)
print(f' R Squared {reg_model.rsquared}')
print(f'Adjusted R Squared {reg_model.rsquared_adj}')
plot_pred(ytrain,reg_fitted,"Prediction with Mulitiple Linear Regression")
graphit(ytrain, ytest, reg_forecast, "Regression Forecast")
#MSE=np.square(np.subtract(ytrain,reg_fitted)).mean()
mse_forc= mean_squared_error(ytest,reg_forecast)
check.loc["Multiple LR"]= [Q.lb_stat.ravel()[0],mse_pred, mse_forc, round(np.var(residual),2),round(np.var(forecast_error),2)]
print(check)
# Check variance ratio
check2=check.copy(deep=True)
check2['variance ratio']=check.iloc[:,-2]/check.iloc[:,-1]
print(check2)

# ARMA process
# Parameter estimation with LM Algorithm
y=df_train['traffic_volume'].values
# global y,N
N=len(df_train)
coef=LM_algorithm(y=y,na=2,nb=0) #LM Algorithm
print(f'Parameters from LM Algorithm:\n {coef}')
covariance_matrix1=cov_teta
print('Covariance Matrix\n',covariance_matrix1)
c1,c2= coef[0]-(2*np.sqrt(covariance_matrix1[0,0])), coef[0]+(2*np.sqrt(covariance_matrix1[0,0]))
c3,c4= coef[1]-(2*np.sqrt(covariance_matrix1[1,1])), coef[1]+(2*np.sqrt(covariance_matrix1[1,1]))
print(f'Confidence interval:\n {c1},{c2}')
print(f'Confidence interval:\n {c3},{c4}')
zero_pole(na=2,nb=0,teta=coef.ravel())

#+=============================statsmodels version 0.10.2================================
# Param estimation with statsmodels version==0.10.2
# model_coef= sm.tsa.ARMA(np.array(y),order=(2,0)).fit(trend='nc',disp=0)
# print(model_coef.summary())
# print_coef(2,0,model_coef)
#+=======================================================================================

# ARMA(2,0)
# predictgion function with known parameters of ARMA(2,0)

prediction= []
teta= coef.ravel()
teta=np.round(teta,4)
y=df_train['traffic_volume'].ravel()
for i in range(len(df_train)):
    if i==0:
        prediction.append(-teta[0]*y[i])
    else:
        prediction.append((-teta[0])*y[i]-teta[1]*y[i-1])
prediction=np.abs(prediction)
residuals= y[1:]-np.array(prediction[:-1])
plot_pred(y,prediction,"Prediction with ARMA(2,0)") #plot of train and prediction
ACF(residuals,50,'Residuals from ARMA(2,0)')
print('Ljung-Box test:')
Q= sm.stats.acorr_ljungbox(residuals, lags=[50],return_df=True)
print(Q)
ACF_PACF(residuals,50)
mse_arma=mean_squared_error(y[1:], np.array(prediction[:-1]))
print(f'MSE Prediction ARMA: {mse_arma}')
# forecast for ARMA
forecast= []
y=df_test['traffic_volume'].ravel()
for i in range(len(df_test)):
    if i==0:
        forecast.append((-teta[0])*y[-1]-teta[1]*y[-2])
    elif i==1:
        forecast.append((-teta[0])*prediction[-1]-teta[1]*y[-1])
    elif i==2:
        forecast.append((-teta[0]) * forecast[i-1] - teta[1] * prediction[-1])
    else:
        forecast.append((-teta[0]) * forecast[i - 1] - teta[1] * forecast[i-2])
forecast=np.abs(forecast)
forecast_error= y[2:]-np.array(forecast[:-2])
print(f"Variance ratio for ARMA test & forecast: {np.var(df_test['traffic_volume'].ravel())/np.var(forecast)}")
mse_arma_f=mean_squared_error(df_test['traffic_volume'].ravel()[1:], np.array(forecast[:-1]))
print(f'MSE Forecast ARMA: {mse_arma_f}')
graphit(df_train['traffic_volume'].values, df_test['traffic_volume'].values, forecast, "ARMA(2,0) Forecast")
#========================

check.loc["ARMA(200)"]= [Q.lb_stat.ravel()[0],mse_arma, mse_arma_f, round(np.var(residuals),2),round(np.var(forecast_error),2)]
print(check)

#===========SARIMA==========
# Trying seasonal differencing of 24
y_dif_24= dif_s(df_train['traffic_volume'],24)
y_dif_24= np.array(y_dif_24)
cal_rolling_mean_var(y_dif_24)
ADF_Cal(y_dif_24)
kpss_test(y_dif_24)
gpac(10,10,y_dif_24,50)


import statsmodels.api as sm
# SARIMA with python package
#SARIMA= (0,1,1,24)
y=df_train['traffic_volume'].ravel()
sarima= sm.tsa.statespace.SARIMAX(y,order=(0,0,0),seasonal_order=(0,1,1,24),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
results=sarima.fit()
print(results.summary())
sarima_train=results.get_prediction(start=0, end=len(df_train['traffic_volume'].ravel()), dynamic=False)
Sarima_pred=sarima_train.predicted_mean
Sarima_residual= df_train['traffic_volume'].ravel()[1:]-np.abs(Sarima_pred[1:-1])
plot_pred(df_train['traffic_volume'].ravel(),np.abs(Sarima_pred),"Prediction with SARIMA(0,1,1)24") #plot of train and prediction
ACF(Sarima_residual,50,'Prediction with SARIMA(0,1,1)24')
print('Ljung-Box test:')
Q= sm.stats.acorr_ljungbox(Sarima_residual, lags=[50],return_df=True)
print(Q)
ACF_PACF(Sarima_residual,50)
mse_sarima=mean_squared_error(df_train['traffic_volume'].ravel()[1:], Sarima_pred[1:-1])
print(f'MSE Prediction SARIMA: {mse_sarima}')


#forecast
sarima_forecast=results.predict(start=len(df_train), end=(len(df_copy2)))
sarima_f_error=df_test['traffic_volume'].values[2:]-np.abs(sarima_forecast[1:-2])
# ACF(sarima_f_error,50,'Forecast with ARIMA(2,0,0)xSARIMA(0,1,1)24')
graphit(df_train['traffic_volume'].values, df_test['traffic_volume'].values, np.abs(sarima_forecast[1:]), "SARIMA Forecast")
mse_sarima_f=mean_squared_error(df_test['traffic_volume'].ravel()[2:], sarima_forecast[1:-2])
print(f'MSE forecast SARIMA: {mse_sarima_f}')
check.loc["SARIMA(0,1,1,24)"]= [Q.lb_stat.ravel()[0],mse_sarima, mse_sarima_f, round(np.var(Sarima_residual),2),round(np.var(sarima_f_error),2)]

#code is seperate in an ipython notebook ran on google colab
check.loc["LSTM"]= [28721.48,1251498.2, 6719125.78, 2001939,6627908]
print(check)

# Check variance ratio
check2=check.copy(deep=True)
check2['variance ratio']=check.iloc[:,-2]/check.iloc[:,-1]
check2['RMSE_Forecast']=np.sqrt(check2.iloc[:,2])
check2.sort_values(by=['RMSE_Forecast','MSE_f','variance ratio'],ascending=True,inplace=True)
print(check2)