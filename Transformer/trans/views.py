from django.shortcuts import render
from django.http import HttpResponse
import matplotlib
import pandas as pd
import plotly.offline as pyoff
import plotly.graph_objs as go
import requests
import datetime
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pmdarima as pm
import seaborn as sn
import matplotlib.pyplot as plt
import io
import urllib,base64
from pmdarima import auto_arima 
import warnings 
import seaborn as sns
import chart_studio.plotly as py
import os
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import plotly.figure_factory as ff
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Create your views here.
def fault(request):
    url="http://182.18.164.20/transformer_api/overview_locations"
    response1=requests.get(url)
    res=response1.json()
    imei=[]
    loc=[]
    msgovf=[]
    msgfrqf=[]
    msgpff=[]
    for i in range(len(res)):
        imei.append(res[i]['DeviceImei'])
        loc.append(res[i]['location'])
        url = "http://182.18.164.20/transformer_api/powerfactor/"+res[i]['DeviceImei']
        user = "admin"
        passwd = "admin@123"
        auth_values = (user, passwd)
        response = requests.get(url, auth=auth_values)
        cv = pd.DataFrame.from_dict(response.json(), orient='columns')
        url = "http://182.18.164.20/transformer_api/overview/"+res[i]['DeviceImei']
        user = "admin"
        passwd = "admin@123"
        auth_values = (user, passwd)
        response = requests.get(url, auth=auth_values)
        ov=pd.DataFrame.from_dict(response.json(), orient='columns')
        ov.isnull().sum()
        ov['DeviceTimeStamp_'] = pd.to_datetime(ov['DeviceTimeStamp'])
        import datetime
        x=ov['DeviceTimeStamp_'].max()
        y=ov['DeviceTimeStamp_'].min()
        days = datetime.timedelta(7)
        week=x-days
        cvlastweek=ov[ov['DeviceTimeStamp_']>=week]
        cvlastweek
        cvlastweek = cvlastweek.iloc[::-1]
        cvlastweek
        cvlastweek=cvlastweek.reset_index()
        cvlastweek
        df=cvlastweek.copy()
        hours=[]
        for i in df['DeviceTimeStamp']:
            ts = i[0:13]+':00:00'
            hours.append(ts)
        df['hours'] = hours
        df = df[['hours', 'OTI', 'ATI', 'OLI', 'OTI_A', 'OTI_T', 'MOG_A']]
        df['OTI']=df['OTI'].astype(float)
        df['ATI']=df['ATI'].astype(float)
        df=df.groupby(['hours']).mean()
        df=df.reset_index()
        df['hours_'] = pd.to_datetime(df['hours'])
        train = df.iloc[:len(df)-24] 
        test = df.iloc[len(df)-24:] 
        from statsmodels.tsa.statespace.sarimax import SARIMAX 
        model = SARIMAX(df['OTI'],order = (2, 0, 0),seasonal_order =(2, 1, 0, 24)) 
        result = model.fit() 

        forecast = result.predict(start = len(df), end = (len(df)-1) + 24,typ = 'levels').rename('Forecast') 
        df_hist = df[['hours_', 'OTI']]
        fc = forecast.to_frame()
        future_time_stamps=[]
        ts=df['hours_'].max()
        for i in range(0, 24):
            ts = ts + datetime.timedelta(minutes = 60)
            future_time_stamps.append(ts)
        fc['hours_'] = np.array(future_time_stamps)
        fault_query = df[df['OTI'] > 100]
        fault_query_future = fc[fc['Forecast'] > 100]
        msgov=''
        if(len(fault_query_future)>0):
            msgov = 'Fault may occur '+str(len(fault_query_future))+' times in the next 24 hours, MOG Alarm might trigger'
        else:
            msgov = 'Fault may not occur in the next 24 hours'
        msgovf.append(msgov)
        cv.isnull().sum()
        cv['DeviceTimeStamp_'] = pd.to_datetime(cv['DeviceTimeStamp'])
        import datetime
        x=cv['DeviceTimeStamp_'].max()
        y=cv['DeviceTimeStamp_'].min()
        days = datetime.timedelta(7)
        week=x-days
        cvlastweek=cv[cv['DeviceTimeStamp_']>=week]
        cvlastweek = cvlastweek.iloc[::-1]
        cvlastweek=cvlastweek.reset_index()
        df=cvlastweek.copy()
        hours=[]
        for i in df['DeviceTimeStamp']:
            ts = i[0:13]+':00:00'
            hours.append(ts)
        df['hours'] = hours
        df=df.groupby(['hours']).mean()
        df=df.reset_index()
        df['hours_'] = pd.to_datetime(df['hours'])
        model = SARIMAX(df['FRQ'],order = (2, 0, 0),seasonal_order =(2, 1, 2, 24)) 
        result = model.fit() 

        forecast = result.predict(start = len(df), end = (len(df)-1) + 24,typ = 'levels').rename('Forecast') 
        df_hist = df[['hours_', 'FRQ']]
        fc = forecast.to_frame()
        import datetime
        future_time_stamps=[]
        ts=df['hours_'].max()
        for i in range(0, 24):
            ts = ts + datetime.timedelta(minutes = 60)
            future_time_stamps.append(ts)
        fc['hours_'] = np.array(future_time_stamps)
        fault_query = df.loc[(df['FRQ']>50.5)|(df['FRQ']<49.5)]
        fault_query_future = fc.loc[(fc['Forecast']>50.5)|(fc['Forecast']<49.5)]
        msgfrq=''
        if(len(fault_query_future)>0):
            msgfrq = 'Fault may occur '+str(len(fault_query_future))+' times in the next 24 hours, OTI ALarm might Trigger'
        else:
            msgfrq = 'Fault may not occur in the next 24 hours'
        msgfrqf.append(msgfrq)
        model = SARIMAX(df['Avg_PF'],order = (1, 0, 0),seasonal_order =(0, 1, 2, 24)) 
        result = model.fit() 
        
        # Forecast for the next 3 years 
        forecast = result.predict(start = len(df), end = (len(df)-1) + 24,typ = 'levels').rename('Forecast')
        df_hist = df[['hours_', 'Avg_PF']]
        fc = forecast.to_frame()
        import datetime
        future_time_stamps=[]
        ts=df['hours_'].max()
        for i in range(0, 24):
            ts = ts + datetime.timedelta(minutes = 60)
            future_time_stamps.append(ts)
        fc['hours_'] = np.array(future_time_stamps)
        fault_query = df[df['Avg_PF'] < 0.9]
        fault_query_future = fc[fc['Forecast'] < 0.9]
        msgpf=''
        if(len(fault_query_future)>0):
            msg = 'Fault may occur '+str(len(fault_query_future))+' times in the next 24 hours, OTI Alarm might trigger'
        else:
            msgpf = 'Fault may not occur in the next 24 hours'
        msgpff.append(msgpf)
        break
    data={'loc':loc,'imei':imei,'msgovf':msgovf,'msgpff':msgpff,'msgfrqf':msgfrqf}
    return render(request,'menu/fault.html',{'data':data,'loc':loc,'imei':imei,'msgovf':msgovf,'msgpff':msgpff,'msgfrqf':msgfrqf}) 
def fault1(request):
    imei=request.GET.get('imei')
    pack=request.GET.get('packettype')
    aid=request.GET.get('parameter')
    url="http://182.18.164.20/transformer_api/overview_locations"
    response1=requests.get(url)
    res=response1.json()
    if(imei!=None and pack!=None and aid!=None and imei!='sd' and pack!='pt' and aid!='pr'):
        for i in range(len(res)):
            if(res[i]['DeviceImei']==imei):
                location=res[i]['location']
                break
        url = "http://182.18.164.20/transformer_api/"+pack+"/"+imei
        user = "admin"
        passwd = "admin@123"
        auth_values = (user, passwd)
        response = requests.get(url, auth=auth_values)
        df = pd.DataFrame.from_dict(response.json(), orient='columns')
        title=aid
        if(pack=="current_voltage"):
            if(aid in ["VL12","VL23","VL31"]):
                df_fault_vl12=df.loc[(df[aid]<390)|(df[aid]>450)]
                df_fault_vl12.head(5)
            elif(aid in ['VL1','VL2','VL3']):
                df_fault_vl12=df.loc[(df[aid]==0)]
                df_fault_vl12.head(5)
        elif(pack=="overview"):
            if(aid=="MOG_A"):
                title='MOG_ALARM'
                df_fault_vl12=df.loc[df['MOG_A']==1]
        elif(pack=="powerfactor"):
            if(aid in ["PFL1","PFL2","PFL3"]):
                df_fault_vl12=df.loc[(df[aid]<0.7)]
                df_fault_vl12.head()

        plot_data = [
                go.Scatter(
                    x=df['DeviceTimeStamp'],
                    y=df[aid],
                    name='Historical '+ title
                ),
                go.Scatter(
                    x=df_fault_vl12['DeviceTimeStamp'],
                    y=df_fault_vl12[aid],
                    mode='markers',
                    name='Alarm',
                    marker= dict(size= 7,
                        line= dict(width=1),
                        color= 'Red',
                        opacity= 0.8
                    )
                    )
        ]

        plot_layout = go.Layout(
                        title=aid+","+pack+', Device Id: '+imei+', fault occured '+str(len(df_fault_vl12))+' times',
                        xaxis_title='Time',
                        yaxis_title=title,
                        plot_bgcolor='rgba(0,0,0,0)',
                        
        )
        fig = go.Figure(data=plot_data, layout=plot_layout)
        #fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="linear"))

        graph_d = pyoff.plot(fig, auto_open=False, output_type="div")
                        


        return render(request,'menu/fault.html',{'graph_div1':graph_d,'devices':response1.json(),'imei':imei,'location':location,'pack':pack,'aid':aid})

    else:
        location1=None
        imei1=None
        pack1=None
        aid1=None
        if(imei!='sd' and imei!=None):
            for i in range(len(res)):
                if(res[i]['DeviceImei']==imei):
                    location1=res[i]['location']
                    break
            imei1=imei
        if(pack!='pt' and pack!=None):
            pack1=pack
        if(aid!='pr' and aid!=None):
            aid1=aid
        return render(request,'menu/fault.html',{'imei':imei1,'pack':pack1,'aid':aid1,'location':location1, 'devices':response1.json()})


# Create your views here.
def index(request):
    
    return render(request,'index.html')

def dashboard(request):
    url = "http://182.18.164.20/transformer_api/Total_Power/"
    user = "admin"
    passwd = "admin@123"
    auth_values = (user, passwd)
    response = requests.get(url, auth=auth_values)
    tp = pd.DataFrame.from_dict(response.json(), orient='columns')
    tp.isnull().sum()
    tp['DeviceTimeStamp_'] = pd.to_datetime(tp['DeviceTimeStamp'])
    tp.head()
    import datetime
    x=tp['DeviceTimeStamp_'].max()
    y=tp['DeviceTimeStamp_'].min()
    days = datetime.timedelta(90)
    month_3=x-days
    tplastweek=tp[tp['DeviceTimeStamp_']>=month_3]
    tplastweek
    tplastweek = tplastweek.iloc[::-1]
    tplastweek
    tplastweek=tplastweek.reset_index()
    tplastweek
    df=tplastweek.copy()
    days=[]
    for i in df['DeviceTimeStamp']:
        ts = i[0:10]
        days.append(ts)
    df['days'] = days
    df
    df = df[['days', 'MPD']]
    df
    df=df.groupby(['days']).mean()
    df
    df=df.reset_index()
    df
    df['days_'] = pd.to_datetime(df['days'])
    df
    train = df.iloc[:len(df)-10] 
    test = df.iloc[len(df)-10:] 
    # Fit a SARIMAX(3, 0, 0)x(2, 1, 0, 100) on the training set 
    from statsmodels.tsa.statespace.sarimax import SARIMAX 
    
    model = SARIMAX(train['MPD'], order = (1, 1, 1), seasonal_order =(2, 1, 0, 30)) 
    
    result = model.fit() 
    result.summary() 
    start = len(train) 
    end = len(train) + len(test) - 1
    predictions = result.predict(start, end, typ = 'levels').rename("Predictions") 

    # plot predictions and actual values 
    predictions.plot(legend = True)
    test['MPD'].plot(legend = True)
    error = mean_squared_error(test['MPD'], predictions)
    print('Test MSE: %.3f' % error)
    rmse_inut = np.sqrt(mean_squared_error(test['MPD'], predictions))
    print('Test RMSE: %.3f' % rmse_inut)
    model = SARIMAX(df['MPD'],order = (1, 1, 1),seasonal_order =(2, 1, 0, 30)) 
    result = model.fit() 
    
    # Forecast for the next 3 years 
    forecast = result.predict(start = len(df), end = (len(df)-1) + 10,typ = 'levels').rename('Forecast') 
    
    # Plot the forecast values 
    df['MPD'].plot(figsize = (12, 5), legend = True) 
    forecast.plot(legend = True) 
    df_hist = df[['days_', 'MPD']]
    df_hist
    fc = forecast.to_frame()
    fc
    import datetime
    future_time_stamps=[]
    ts=df['days_'].max()
    for i in range(0, 10):
        ts = ts + datetime.timedelta(days = 1)
        future_time_stamps.append(ts)
    future_time_stamps
    fc['days_'] = np.array(future_time_stamps)
    fc['Total demand'] = fc['Forecast']*52 
    plot_data = [
        go.Scatter(
            x=df['days_'],
            y=df['MPD'],
            name='Historical MPD'
        ),
        go.Scatter(
            x=fc['days_'],
            y=fc['Forecast'],
            name='Forecast'
        )
    ]
    plot_layout = go.Layout(
            title='Total Average Demand',
            xaxis_title='Time',
            yaxis_title='MPD',
            plot_bgcolor='rgba(0,0,0,0)'
        )
    fig = go.Figure(data=plot_data, layout=plot_layout)

    graph_d = pyoff.plot(fig, auto_open=False, output_type="div")
    return render(request,'menu/dashboard.html',{'graph':graph_d})

def forecasting(request):
    imei=request.GET.get('imei')
    pack=request.GET.get('packettype')
    aid=request.GET.get('parameter')
    method=request.GET.get('method')
    url="http://182.18.164.20/transformer_api/overview_locations"
    response1=requests.get(url)
    res=response1.json()
    if(imei!=None and pack!=None and aid!=None and imei!='sd' and pack!='pt' and aid!='pr' and method!='m' and method!=None):
        for i in range(len(res)):
            if(res[i]['DeviceImei']==imei):
                location=res[i]['location']
                break
        url = "http://182.18.164.20/transformer_api/"+pack+"/"+imei
        user = "admin"
        passwd = "admin@123"
        auth_values = (user, passwd)
        response = requests.get(url, auth=auth_values)
        cv = pd.DataFrame.from_dict(response.json(), orient='columns')
        cv.isnull().sum()
        tp=cv
        ov=cv
        if(method=="Smoothing"):
                if(aid == 'OTI' or aid== 'ATI'):
                    cv['OTI']=cv['OTI'].astype(float)
                    cv['ATI']=cv['ATI'].astype(float)

                cv['DeviceTimeStamp_'] = pd.to_datetime(cv['DeviceTimeStamp'])
                x=cv['DeviceTimeStamp_'].max()
                y=cv['DeviceTimeStamp_'].min()
                days = datetime.timedelta(7)
                week=x-days
                cvlastweek=cv[cv['DeviceTimeStamp_']>=week]
                cvlastweek = cvlastweek.iloc[::-1]
                cvlastweek=cvlastweek.reset_index()

                d=cvlastweek[aid]
                ax=d.plot()
                ax.set_xlabel("time")
                ax.set_ylabel(aid)
                
                future_time_stamps=[]
                ts=x
                for i in range(0, 100):
                    ts = ts + datetime.timedelta(minutes = 15)
                    future_time_stamps.append(ts)

                fit1 = SimpleExpSmoothing(d, initialization_method="heuristic").fit(smoothing_level=0.2,optimized=False)
                fcast1 = fit1.forecast(100).rename(r'$\alpha=0.2$')
                fit2 = SimpleExpSmoothing(d, initialization_method="heuristic").fit(smoothing_level=0.6,optimized=False)
                fcast2 = fit2.forecast(100).rename(r'$\alpha=0.6$')
                fit3 = SimpleExpSmoothing(d, initialization_method="estimated").fit()
                fcast3 = fit3.forecast(100).rename(r'$\alpha=%s$'%fit3.model.params['smoothing_level'])

                import plotly.graph_objs as go
                plot_data = [
                    go.Scatter(
                        x=cvlastweek['DeviceTimeStamp_'],
                        y=cvlastweek[aid],
                        name='Historical '+ aid
                    ),
                    go.Scatter(
                        x=future_time_stamps,
                        y=fcast1,
                        name='Forecast(alpha=0.2)'
                    ),
                    go.Scatter(
                        x=future_time_stamps,
                        y=fcast2,
                        name='Forecast(alpha=0.6)'
                    ),
                    go.Scatter(
                        x=future_time_stamps,
                        y=fcast3,
                        name='Forecast(optimized)'
                    )
                        
                ]
                plot_layout=go.Layout(
                        title="Simple smoothing, " + aid +", "+pack+', Device Id: '+imei,
                        xaxis_title='Time',
                        yaxis_title=aid,
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                fig = go.Figure(data=plot_data, layout=plot_layout)

                graph_d = pyoff.plot(fig, auto_open=False, output_type="div") 
                if len(set(d)) == 1:
                    plot_data = [
                        go.Scatter(
                            x=cvlastweek['DeviceTimeStamp_'],
                            y=cvlastweek[aid],
                            name=aid
                        )

                    ]
                    plot_layout=go.Layout(
                            title=aid+" is constant and cannot be forcasted by Holt Winter's",
                            xaxis_title='Time',
                            yaxis_title=aid,
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                    fig = go.Figure(data=plot_data, layout=plot_layout)
                if len(set(d)) == 1:
                    plot_data = [
                        go.Scatter(
                            x=cvlastweek['DeviceTimeStamp_'],
                            y=cvlastweek[aid],
                            name=aid
                        )

                    ]
                    plot_layout=go.Layout(
                            title=aid+" is constant and cannot be forcasted by Holt Winter's",
                            xaxis_title='Time',
                            yaxis_title=aid,
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                    fig = go.Figure(data=plot_data, layout=plot_layout)
                else:
                    d=np.absolute(d)
                    fit1 = ExponentialSmoothing(np.absolute(d), seasonal_periods=100, trend='add', seasonal='add', use_boxcox=True, initialization_method="estimated").fit()
                    fit2 = ExponentialSmoothing(d, seasonal_periods=100, trend='add', seasonal='mul', use_boxcox=True, initialization_method="estimated").fit()
                    fit3 = ExponentialSmoothing(d, seasonal_periods=100, trend='add', seasonal='add', damped_trend=True, use_boxcox=True, initialization_method="estimated").fit()
                    fit4 = ExponentialSmoothing(d, seasonal_periods=100, trend='add', seasonal='mul', damped_trend=True, use_boxcox=True, initialization_method="estimated").fit()
                    results=pd.DataFrame(index=[r"$\alpha$",r"$\beta$",r"$\phi$",r"$\gamma$",r"$l_0$","$b_0$","SSE"])
                    params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'smoothing_seasonal', 'initial_level', 'initial_trend']
                    results["Additive"]       = [fit1.params[p] for p in params] + [fit1.sse]
                    results["Multiplicative"] = [fit2.params[p] for p in params] + [fit2.sse]
                    results["Additive Dam"]   = [fit3.params[p] for p in params] + [fit3.sse]
                    results["Multiplica Dam"] = [fit4.params[p] for p in params] + [fit4.sse]


                    fcast1=fit1.forecast(100).rename('Holt-Winters (add-add-seasonal)')
                    print(fcast1)
                    fcast2=fit2.forecast(100).rename('Holt-Winters (add-mul-seasonal)')
                    plot_data = [
                        go.Scatter(
                            x=cvlastweek['DeviceTimeStamp_'],
                            y=cvlastweek[aid],
                            name='Historical ' + aid
                        ),
                        go.Scatter(
                            x=future_time_stamps,
                            y=fcast1,
                            name='Holt-Winters (add-add-seasonal)Forecast'
                        ),
                        go.Scatter(
                            x=future_time_stamps,
                            y=fcast2,
                            name='Holt-Winters (add-mul-seasonal)Forecast'
                        )
                    ]
                    plot_layout=go.Layout(
                            title="Exponential smoothing, " + aid +", "+pack+', Device Id: '+imei,
                            xaxis_title='Time',
                            yaxis_title=aid,
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                    fig = go.Figure(data=plot_data, layout=plot_layout)
                graph_d1 = pyoff.plot(fig, auto_open=False, output_type="div") 
                return render(request,'menu/forecasting.html',{'method':method,'location':location,'pack':pack,'imei':imei,'aid':aid,'graph_div1':graph_d,'graph_div2':graph_d1,'devices':response1.json() })
        elif(method=="Sarima"):
            if(pack=="current_voltage"):
                cv = pd.DataFrame.from_dict(response.json(), orient='columns')
                cv.isnull().sum()
                cv['DeviceTimeStamp_'] = pd.to_datetime(cv['DeviceTimeStamp'])
                x=cv['DeviceTimeStamp_'].max()
                y=cv['DeviceTimeStamp_'].min()
                days = datetime.timedelta(7)
                week=x-days
                cvlastweek=cv[cv['DeviceTimeStamp_']>=week]
                cvlastweek = cvlastweek.iloc[::-1]
                cvlastweek=cvlastweek.reset_index()

                df=cvlastweek.copy()
                df['DeviceTimeStamp']
                hours=[]
                for i in df['DeviceTimeStamp']:
                    ts = i[0:13]+':00:00'
                    hours.append(ts)
                    print(ts)
                df['hours'] = hours
                df = df[['hours', 'VL1', 'VL2', 'VL3', 'IL1', 'IL2', 'IL3', 'VL12', 'VL23', 'VL31', 'AVL', 'INUT']]
                df=df.groupby(['hours']).mean()
                df=df.reset_index()
                df['hours_'] = pd.to_datetime(df['hours'])
                train = df.iloc[:len(df)-24] 
                test = df.iloc[len(df)-24:] 
                from statsmodels.tsa.statespace.sarimax import SARIMAX 
                
                if(aid in ['IL1','IL2','IL3']):
                        model = SARIMAX(train[aid], order = (3, 0, 1), seasonal_order =(0, 1, 2, 24)) 
                elif(aid in ['VL1','VL2','VL3']):
                        model = SARIMAX(train['VL1'], order = (2, 0, 0), seasonal_order =(2, 1, 0, 24))
                
                result = model.fit() 
                result.summary() 
                start = len(train) 
                end = len(train) + len(test) - 1
                start = len(train) 
                end = len(train) + len(test) - 1
                
                # Predictions for one-year against the test set 
                predictions = result.predict(start, end, typ = 'levels').rename("Predictions") 

                # plot predictions and actual values 
                predictions.plot(legend = True)
                test[aid].plot(legend = True)
                error = mean_squared_error(test[aid], predictions)
                print('Test MSE: %.3f' % error)
                rmse_inut = np.sqrt(mean_squared_error(test[aid], predictions))
                print('Test RMSE: %.3f' % rmse_inut)
                if(aid in ['IL1','IL2','IL3']):
                        model = SARIMAX(df[aid],order = (3, 0, 1),seasonal_order =(0, 1, 2, 24)) 
                elif(aid in ['VL1','VL2','VL3']):
                        model = SARIMAX(train['VL1'], order = (2, 0, 0), seasonal_order =(2, 1, 0, 24))
                result = model.fit() 
                
                # Forecast for the next 3 years 
                forecast = result.predict(start = len(df), end = (len(df)-1) + 24,typ = 'levels').rename('Forecast') 
                
                # Plot the forecast values 
                df[aid].plot(figsize = (12, 5), legend = True) 
                forecast.plot(legend = True) 
                df_hist = df[['hours_', aid]]
                fc = forecast.to_frame()
                future_time_stamps=[]
                ts=df['hours_'].max()
                for i in range(0, 24):
                    ts = ts + datetime.timedelta(minutes = 60)
                    future_time_stamps.append(ts)
                future_time_stamps
                fc['hours_'] = np.array(future_time_stamps)
                if(aid in ['IL1','IL2','IL3']):
                    fault_query = df[df[aid] > 180]
                    fault_query_future = fc[fc['Forecast'] > 180]
                elif(aid in ['VL1','VL2','VL3']):
                    print(True)
                    fault_query=df.loc[(df[aid]>264)|(df[aid]<195.5)]
                    fault_query_future =  fc.loc[(fc['Forecast']>264)|(fc['Forecast']<195.5)]
                msg=''
                if(len(fault_query_future)>0):
                    msg = 'Fault may occur '+str(len(fault_query_future))+' times in the next 24 hours'
                else:
                    msg = 'Fault may not occur in the next 24 hours'
                if(aid in ['VL1','VL2','VL3']):
                        print("true")
                        fault_query = df.loc[(df[aid]>264)|(df[aid]<195.5)]
                        
                        if(len(fault_query_future)>0):
                            msg = 'Fault may occur '+str(len(fault_query_future))+' times in the next 24 hours'
                        else:
                            msg = 'Fault may not occur in the next 24 hours'
                import plotly.graph_objs as go
                plot_data = [
                    go.Scatter(
                        x=df['hours_'],
                        y=df[aid],
                        name='Historical '+aid
                    ),
                    go.Scatter(
                        x=fc['hours_'],
                        y=fc['Forecast'],
                        name='Forecast'
                    ),
                    go.Scatter(
                        x=fault_query['hours_'],
                        y=fault_query[aid],
                        mode='markers',
                        name='Alarm',
                        marker= dict(size= 7,
                            line= dict(width=1),
                            color= 'Blue',
                            opacity= 0.8
                        )
                    ),
                    go.Scatter(
                        x=fault_query_future['hours_'],
                        y=fault_query_future['Forecast'],
                        mode='markers',
                        name='Predicted Alarm',
                        marker= dict(size= 7,
                            line= dict(width=1),
                            color= 'red',
                            opacity= 0.8
                        )
                    )
                ]
                plot_layout = go.Layout(
                        title="Sarimax, " + aid +", "+pack+', Device Id: '+imei +' '+msg,
                        xaxis_title='Time',
                        yaxis_title=aid,
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                
                fig = go.Figure(data=plot_data, layout=plot_layout)
            elif(pack=="Total_Power"):
                if(aid=="KVA" or aid=="KW"):
                    tp.isnull().sum()
                    tp['DeviceTimeStamp_'] = pd.to_datetime(tp['DeviceTimeStamp'])
                    x=tp['DeviceTimeStamp_'].max()
                    y=tp['DeviceTimeStamp_'].min()
                    days = datetime.timedelta(7)
                    week=x-days
                    cvlastweek=tp[tp['DeviceTimeStamp_']>=week]
                    cvlastweek = cvlastweek.iloc[::-1]
                    cvlastweek=cvlastweek.reset_index()
                    df=cvlastweek.copy()
                    hours=[]
                    for i in df['DeviceTimeStamp']:
                        ts = i[0:13]+':00:00'
                        hours.append(ts)
                        print(ts)
                    df['hours'] = hours
                    df = df[['hours', 'KWH', 'KVARH', 'Sum_I', 'KW', 'KVA', 'KVAR']]
                    df=df.groupby(['hours']).mean()
                    df=df.reset_index()
                    df['hours_'] = pd.to_datetime(df['hours'])
                    # Split data into train / test sets 
                    train = df.iloc[:len(df)-24] 
                    test = df.iloc[len(df)-24:] 
                    # Fit a SARIMAX(3, 0, 0)x(2, 1, 0, 100) on the training set 
                    from statsmodels.tsa.statespace.sarimax import SARIMAX 
                    model = SARIMAX(train[aid], order = (1, 0, 0), seasonal_order =(2, 1, 0, 24)) 
                    
                    result = model.fit() 
                    result.summary() 
                    start = len(train) 
                    end = len(train) + len(test) - 1
                    
                    # Predictions for one-year against the test set 
                    predictions = result.predict(start, end, typ = 'levels').rename("Predictions") 

                    # plot predictions and actual values 
                    predictions.plot(legend = True)
                    test[aid].plot(legend = True)
                    # Train the model on the full dataset 
                    model = SARIMAX(df[aid],order = (1, 0, 0),seasonal_order =(2, 1, 0, 24)) 
                    result = model.fit() 
                    
                    # Forecast for the next 3 years 
                    forecast = result.predict(start = len(df), end = (len(df)-1) + 24,typ = 'levels').rename('Forecast') 
                    
                    # Plot the forecast values 
                    df[aid].plot(figsize = (12, 5), legend = True) 
                    forecast.plot(legend = True) 
                    df_hist = df[['hours_', aid]]
                    fc = forecast.to_frame()
                    future_time_stamps=[]
                    ts=df['hours_'].max()
                    for i in range(0, 24):
                        ts = ts + datetime.timedelta(minutes = 60)
                        future_time_stamps.append(ts)
                    future_time_stamps
                    fc['hours_'] = np.array(future_time_stamps)
                    import plotly.graph_objs as go
                    plot_data = [
                        go.Scatter(
                            x=df['hours_'],
                            y=df[aid],
                            name='Historical '+aid
                        ),
                        go.Scatter(
                            x=fc['hours_'],
                            y=fc['Forecast'],
                            name='Forecast'
                        )

                    ]
                    plot_layout = go.Layout(
                            title='Load, Device Id: '+imei,
                            xaxis_title='Time',
                            yaxis_title=aid,
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                    fig = go.Figure(data=plot_data, layout=plot_layout)
            
                else:
                    tp.isnull().sum()
                    tp['DeviceTimeStamp_'] = pd.to_datetime(tp['DeviceTimeStamp'])
                    x=tp['DeviceTimeStamp_'].max()
                    y=tp['DeviceTimeStamp_'].min()
                    days = datetime.timedelta(7)
                    week=x-days
                    cvlastweek=tp[tp['DeviceTimeStamp_']>=week]
                    cvlastweek = cvlastweek.iloc[::-1]
                    cvlastweek=cvlastweek.reset_index()
                    df=cvlastweek.copy()
                    hours=[]
                    for i in df['DeviceTimeStamp']:
                        ts = i[0:13]+':00:00'
                        hours.append(ts)
                        print(ts)
                    df['hours'] = hours
                    df = df[['hours', 'KWH', 'KVARH', 'Sum_I', 'KW', 'KVA', 'KVAR']]
                    df=df.groupby(['hours']).mean()
                    df=df.reset_index()
                    df['hours_'] = pd.to_datetime(df['hours'])
                    # Split data into train / test sets 
                    train = df.iloc[:len(df)-24] 
                    test = df.iloc[len(df)-24:] 
                    # Fit a SARIMAX(3, 0, 0)x(2, 1, 0, 100) on the training set 
                    from statsmodels.tsa.statespace.sarimax import SARIMAX
                    model = SARIMAX(train[aid], order = (0, 0, 1), seasonal_order =(2, 1, 2, 24))
                    
                    result = model.fit() 
                    result.summary() 

                    start = len(train) 
                    end = len(train) + len(test) - 1
                    
                    # Predictions for one-year against the test set 
                    predictions = result.predict(start, end, typ = 'levels').rename("Predictions") 

                    # plot predictions and actual values 
                    predictions.plot(legend = True)
                    test[aid].plot(legend = True)

                    error = mean_squared_error(test[aid], predictions)
                    print('Test MSE: %.3f' % error)
                    rmse_inut = np.sqrt(mean_squared_error(test[aid], predictions))
                    print('Test RMSE: %.3f' % rmse_inut)
                    # Train the model on the full dataset 
                    model = SARIMAX(train[aid], order = (0, 0, 1), seasonal_order =(2, 1, 2, 24))
                    result = model.fit() 
                    
                    # Forecast for the next 3 years 
                    forecast = result.predict(start = len(df), end = (len(df)-1) + 24,typ = 'levels').rename('Forecast') 
                    
                    # Plot the forecast values 
                    df[aid].plot(figsize = (12, 5), legend = True) 
                    forecast.plot(legend = True) 
                    df_hist = df[['hours_', aid]]
                    fc = forecast.to_frame()
                    future_time_stamps=[]
                    ts=df['hours_'].max()
                    for i in range(0, 24):
                        ts = ts + datetime.timedelta(minutes = 60)
                        future_time_stamps.append(ts)
                    future_time_stamps
                    msg=''
                    fc['hours_'] = np.array(future_time_stamps)
                   
                    import plotly.graph_objs as go
                    plot_data = [
                        go.Scatter(
                            x=df['hours_'],
                            y=df[aid],
                            name='Historical '+aid
                        ),
                        go.Scatter(
                            x=fc['hours_'],
                            y=fc['Forecast'],
                            name='Forecast'
                        )

                    ]
                    plot_layout = go.Layout(
                            title=pack+', '+ 'Device Id: '+imei+msg,
                            xaxis_title='Time',
                            yaxis_title=aid,
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                    fig = go.Figure(data=plot_data, layout=plot_layout)
            elif(pack=="overview"):
                ov.isnull().sum()
                ov['DeviceTimeStamp_'] = pd.to_datetime(ov['DeviceTimeStamp'])
                x=ov['DeviceTimeStamp_'].max()
                y=ov['DeviceTimeStamp_'].min()
                days = datetime.timedelta(7)
                week=x-days
                cvlastweek=ov[ov['DeviceTimeStamp_']>=week]
                cvlastweek = cvlastweek.iloc[::-1]
                cvlastweek=cvlastweek.reset_index()
                df=cvlastweek.copy()
                hours=[]
                for i in df['DeviceTimeStamp']:
                    ts = i[0:13]+':00:00'
                    hours.append(ts)
                    print(ts)
                df['hours'] = hours
                df = df[['hours', 'OTI', 'ATI', 'OLI', 'OTI_A', 'OTI_T', 'MOG_A']]
                df['OTI']=df['OTI'].astype(float)
                df['ATI']=df['ATI'].astype(float)
                df=df.groupby(['hours']).mean()
                df=df.reset_index()
                df['hours_'] = pd.to_datetime(df['hours'])
                train = df.iloc[:len(df)-24] 
                test = df.iloc[len(df)-24:] 
                from statsmodels.tsa.statespace.sarimax import SARIMAX 
                
                model = SARIMAX(train[aid], order = (2, 0, 0), seasonal_order =(2, 1, 0, 24)) 
                
                result = model.fit() 
                result.summary() 
                start = len(train) 
                end = len(train) + len(test) - 1

                start = len(train) 
                end = len(train) + len(test) - 1
                
                # Predictions for one-year against the test set 
                predictions = result.predict(start, end, typ = 'levels').rename("Predictions") 

                # plot predictions and actual values 
                predictions.plot(legend = True)
                test[aid].plot(legend = True)

                error = mean_squared_error(test[aid], predictions)
                print('Test MSE: %.3f' % error)
                rmse_inut = np.sqrt(mean_squared_error(test[aid], predictions))
                print('Test RMSE: %.3f' % rmse_inut)

                # Train the model on the full dataset 
                model = SARIMAX(df[aid],order = (2, 0, 0),seasonal_order =(2, 1, 0, 24)) 
                result = model.fit() 
                
                # Forecast for the next 3 years 
                forecast = result.predict(start = len(df), end = (len(df)-1) + 24,typ = 'levels').rename('Forecast') 

                df_hist = df[['hours_', aid]]
                fc = forecast.to_frame()

                future_time_stamps=[]
                ts=df['hours_'].max()
                for i in range(0, 24):
                    ts = ts + datetime.timedelta(minutes = 60)
                    future_time_stamps.append(ts)

                fc['hours_'] = np.array(future_time_stamps)

                fault_query = df[df[aid] > 100]
                fault_query_future = fc[fc['Forecast'] > 100]

                msg=''
                if(len(fault_query_future)>0):
                    msg = 'Fault may occur '+str(len(fault_query_future))+' times in the next 24 hours'
                else:
                    msg = 'Fault may not occur in the next 24 hours'
                import plotly.graph_objs as go
                plot_data = [
                    go.Scatter(
                        x=df['hours_'],
                        y=df[aid],
                        name='Historical '+aid
                    ),
                    go.Scatter(
                        x=fc['hours_'],
                        y=fc['Forecast'],
                        name='Forecast'
                    ),
                    go.Scatter(
                        x=fault_query['hours_'],
                        y=fault_query[aid],
                        mode='markers',
                        name='Alarm',
                        marker= dict(size= 7,
                            line= dict(width=1),
                            color= 'Blue',
                            opacity= 0.8
                        )
                    ),
                    go.Scatter(
                        x=fault_query_future['hours_'],
                        y=fault_query_future['Forecast'],
                        mode='markers',
                        name='Predicted Alarm',
                        marker= dict(size= 7,
                            line= dict(width=1),
                            color= 'red',
                            opacity= 0.8
                        )
                    )
                ]
                plot_layout = go.Layout(
                        title='Overview, Device Id:  '+imei+', '+str(msg),
                        xaxis_title='Time',
                        yaxis_title=aid,
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                fig = go.Figure(data=plot_data, layout=plot_layout)    
                    


            graph = pyoff.plot(fig, auto_open=False, output_type="div") 
            return render(request,'menu/forecasting.html',{'method':method,'location':location,'pack':pack,'imei':imei,'graph':graph,'aid':aid,'devices':response1.json() })
    else:
        location1=None
        imei1=None
        pack1=None
        aid1=None
        if(imei!='sd' and imei!=None):
            for i in range(len(res)):
                if(res[i]['DeviceImei']==imei):
                    location1=res[i]['location']
                    break
            imei1=imei
        if(pack!='pt' and pack!=None):
            pack1=pack
        if(aid!='pr' and aid!=None):
            aid1=aid
        return render(request,'menu/forecasting.html',{'imei':imei1,'pack':pack1,'aid':aid1,'location':location1, 'devices':response1.json()})

def dataanalysis1(request): 
    imei=request.GET.get('imei')
    pack=request.GET.get('packettype')
    url="http://182.18.164.20/transformer_api/overview_locations"
    response1=requests.get(url)
    res=response1.json()
    if(imei!=None and pack!=None  and imei!='sd' and pack!='pt'):
        for i in range(len(res)):
            if(res[i]['DeviceImei']==imei):
                location=res[i]['location']
                break
        url = "http://182.18.164.20/transformer_api/"+pack+"/"+imei
        user = "admin"
        passwd = "admin@123"
        auth_values = (user, passwd)
        response = requests.get(url, auth=auth_values)
        ov = pd.DataFrame.from_dict(response.json(), orient='columns')
        if(pack=="Total_Power"):
            ov_1 = ov[['KWH','KVARH','KW','KVA','KVAR','MPD','MKVAD']].copy()
            coMa=ov_1.corr()
            arr = np.array(coMa)
            np_round = np.around(arr, 3)
            np_round1=np_round
            np_round1=np.nan_to_num(np_round1)
            cols = list(coMa.columns)
            z = np_round1

            x = cols
            y = cols

            z_text = arr

            fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=np_round, colorscale='Viridis')
            fig.update_layout(
            autosize=False,
            width=800,
            height=600)
            graph_d = pyoff.plot(fig, auto_open=False, output_type="div")
    
        elif(pack=="overview"):
            df = pd.DataFrame(ov,columns=['OTI','WTI','ATI','OLI','OTI_A','OTI_T','WTI_A','WTI_T','GOR_A','GOR_T','MOG_A'])
            df['OTI']=df['OTI'].astype(float)
            df['WTI']=df['WTI'].astype(float)
            df['ATI']=df['ATI'].astype(float)
            df['OLI']=df['OLI'].astype(float)
            corrMatrix = df.corr()
            arr = np.array(corrMatrix)
            np_round = np.around(arr, 3)
            np_round1=np_round.copy()
            where_are_NaNs = np.isnan(np_round1)
            np_round1[where_are_NaNs] = 0
            cols = list(corrMatrix.columns)
            z = np_round1

            x = cols
            y = cols

            z_text = arr

            fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=np_round, colorscale='Viridis')
            fig.update_layout(
            autosize=False,
            width=800,
            height=600)
            graph_d = pyoff.plot(fig, auto_open=False, output_type="div")
             


        return render(request,'menu/dataanalysis1.html',{'data1':graph_d,'imei':imei,'location':location,'pack':pack,'devices':response1.json() })
    
    else:
        location1=None
        imei1=None
        pack1=None
        if(imei!='sd' and imei!=None):
            for i in range(len(res)):
                if(res[i]['DeviceImei']==imei):
                    location1=res[i]['location']
                    break
            imei1=imei
        if(pack!='pt' and pack!=None):
            pack1=pack
        return render(request,'menu/dataanalysis1.html',{'imei':imei1,'pack':pack1,'location':location1,'devices':response1.json()})

def dataanalysis2(request):
    imei=request.GET.get('imei')
    pack=request.GET.get('packettype')
    url="http://182.18.164.20/transformer_api/overview_locations"
    response1=requests.get(url)
    res=response1.json()
    
    if(imei!=None and pack!=None  and imei!='sd' and pack!='pt'):
        for i in range(len(res)):
            if(res[i]['DeviceImei']==imei):
                location=res[i]['location']
                break
        url = "http://182.18.164.20/transformer_api/"+pack+"/"+imei
        user = "admin"
        passwd = "admin@123"
        auth_values = (user, passwd)
        response = requests.get(url, auth=auth_values)
        df=pd.DataFrame.from_dict(response.json(),orient='columns')
        if(pack=="current_voltage"):
            df.corr()
            plot_data = [
                go.Scatter(
                    x=df['DeviceTimeStamp'],
                    y=df['VL1'],
                    name='VL1'
                ),
                go.Scatter(
                    x=df['DeviceTimeStamp'],
                    y=df['VL2'],
                    name='VL2'
                )
                
            ]
            plot_layout = go.Layout(
                    title='Voltage',
                    xaxis_title='Time'
                )
            fig = go.Figure(data=plot_data, layout=plot_layout)
            
            plot_data = [
                go.Scatter(
                    x=df['DeviceTimeStamp'],
                    y=df['IL1'],
                    name='IL1'
                ),
                go.Scatter(
                    x=df['DeviceTimeStamp'],
                    y=df['IL2'],
                    name='IL2'
                ),
                go.Scatter(
                    x=df['DeviceTimeStamp'],
                    y=df['IL3'],
                    name='IL3'
                )
                
            ]
            plot_layout = go.Layout(
                    title='Current',
                    xaxis_title='Time'
                )
            fig2 = go.Figure(data=plot_data, layout=plot_layout)
            #fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="linear"))

            plot_data = [
                go.Scatter(
                    x=df['DeviceTimeStamp'],
                    y=df['INUT'],
                    name='INUT'
                )
                
            ]
            plot_layout = go.Layout(
                    title='INUT',
                    xaxis_title='Time'
                )
            fig3 = go.Figure(data=plot_data, layout=plot_layout)
            #fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="linear"))
            graph_d = pyoff.plot(fig, auto_open=False, output_type="div")
            graph_d1 = pyoff.plot(fig2, auto_open=False, output_type="div")
            graph_d2 = pyoff.plot(fig3, auto_open=False, output_type="div")   
            return render(request,'menu/dataanalysis2.html',{'imei':imei,'location':location,'pack':pack,'graph_div3':graph_d2,'graph_div1':graph_d,'graph_div2':graph_d1,'devices':response1.json(),'imei':imei,'pack':pack })
        
        elif(pack=="power"):
            df.head(5)
            df.corr()
            plot_data = [
                go.Scatter(
                    x=df['DeviceTimeStamp'],
                    y=df['WL1'],
                    name='WL1'
                ),
                go.Scatter(
                    x=df['DeviceTimeStamp'],
                    y=df['WL2'],
                    name='WL2'
                ),
                go.Scatter(
                    x=df['DeviceTimeStamp'],
                    y=df['WL3'],
                    name='WL3'
                )
                
            ]
            plot_layout = go.Layout(
                    
                    xaxis_title='Time'
                )
            fig = go.Figure(data=plot_data, layout=plot_layout)
            
            plot_data = [
                go.Scatter(
                    x=df['DeviceTimeStamp'],
                    y=df['VAL1'],
                    name='VAL1'
                ),
                go.Scatter(
                    x=df['DeviceTimeStamp'],
                    y=df['VAL2'],
                    name='VAL2'
                ),
                go.Scatter(
                    x=df['DeviceTimeStamp'],
                    y=df['VAL3'],
                    name='VAL3'
                )
                
            ]
            plot_layout = go.Layout(
                    
                    xaxis_title='Time'
                )
            fig2 = go.Figure(data=plot_data, layout=plot_layout)
            
            plot_data = [
                go.Scatter(
                    x=df['DeviceTimeStamp'],
                    y=df['RVAL1'],
                    name='RVAL1'
                ),
                go.Scatter(
                    x=df['DeviceTimeStamp'],
                    y=df['RVAL2'],
                    name='RVAL2'
                ),
                go.Scatter(
                    x=df['DeviceTimeStamp'],
                    y=df['RVAL3'],
                    name='RVAL3'
                )
                
            ]
            plot_layout = go.Layout(
                    
                    xaxis_title='Time'
                )
            fig3 = go.Figure(data=plot_data, layout=plot_layout)
            graph_d = pyoff.plot(fig, auto_open=False, output_type="div")
            graph_d1 = pyoff.plot(fig2, auto_open=False, output_type="div")
            graph_d2 = pyoff.plot(fig3, auto_open=False, output_type="div")   
            return render(request,'menu/dataanalysis2.html',{'graph_div3':graph_d2,'graph_div1':graph_d,'graph_div2':graph_d1,'devices':response1.json(),'imei':imei,'pack':pack, 'location':location })
        elif(pack=="powerfactor"):
            df.head(5)
            df.corr()
            plot_data = [
                go.Scatter(
                    x=df['DeviceTimeStamp'],
                    y=df['PFL1'],
                    name='PFL1'
                ),
                go.Scatter(
                    x=df['DeviceTimeStamp'],
                    y=df['PFL2'],
                    name='PFL2'
                ),
                go.Scatter(
                    x=df['DeviceTimeStamp'],
                    y=df['PFL3'],
                    name='PFL3'
                )
                
            ]
            plot_layout = go.Layout(
                    
                    xaxis_title='Time'
                )
            fig = go.Figure(data=plot_data, layout=plot_layout)

            plot_data = [
                go.Scatter(
                    x=df['DeviceTimeStamp'],
                    y=df['THDVL1'],
                    name='THDVL1'
                ),
                go.Scatter(
                    x=df['DeviceTimeStamp'],
                    y=df['THDVL2'],
                    name='THDVL2'
                ),
                go.Scatter(
                    x=df['DeviceTimeStamp'],
                    y=df['THDVL3'],
                    name='THDVL3'
                )
                
            ]
            plot_layout = go.Layout(
                    
                    xaxis_title='Time'
                )
            fig2 = go.Figure(data=plot_data, layout=plot_layout)

            plot_data = [
                go.Scatter(
                    x=df['DeviceTimeStamp'],
                    y=df['THDIL1'],
                    name='THDIL1'
                ),
                go.Scatter(
                    x=df['DeviceTimeStamp'],
                    y=df['THDIL2'],
                    name='THDIL2'
                ),
                go.Scatter(
                    x=df['DeviceTimeStamp'],
                    y=df['THDIL3'],
                    name='THDIL3'
                )
                
            ]
            plot_layout = go.Layout(
                    
                    xaxis_title='Time'
                )
            fig3 = go.Figure(data=plot_data, layout=plot_layout)

            plot_data = [
                go.Scatter(
                    x=df['DeviceTimeStamp'],
                    y=df['MDIL1'],
                    name='MDIL1'
                ),
                go.Scatter(
                    x=df['DeviceTimeStamp'],
                    y=df['MDIL2'],
                    name='MDIL2'
                ),
                go.Scatter(
                    x=df['DeviceTimeStamp'],
                    y=df['MDIL3'],
                    name='MDIL3'
                )
                
            ]
            plot_layout = go.Layout(
                    
                    xaxis_title='Time'
                )
            fig4 = go.Figure(data=plot_data, layout=plot_layout)

            plot_data = [
                go.Scatter(
                    x=df['DeviceTimeStamp'],
                    y=df['FRQ'],
                    name='FRQ'
                )
                
                
            ]
            plot_layout = go.Layout(
                title="PRQ",
                    
                    xaxis_title='Time'
                )
            fig5 = go.Figure(data=plot_data, layout=plot_layout)

            plot_data = [
                go.Scatter(
                    x=df['DeviceTimeStamp'],
                    y=df['Avg_PF'],
                    name='Avg_PF'
                ),
                go.Scatter(
                    x=df['DeviceTimeStamp'],
                    y=df['Sum_PF'],
                    name='Sum_PF'
                )
                
            ]
            plot_layout = go.Layout(
                    
                    xaxis_title='Time'
                )
            fig6 = go.Figure(data=plot_data, layout=plot_layout)
            graph_d = pyoff.plot(fig, auto_open=False, output_type="div")
            graph_d1 = pyoff.plot(fig2, auto_open=False, output_type="div")
            graph_d2 = pyoff.plot(fig3, auto_open=False, output_type="div")   
            graph_d3 = pyoff.plot(fig4, auto_open=False, output_type="div")
            graph_d4 = pyoff.plot(fig5, auto_open=False, output_type="div")
            graph_d5 = pyoff.plot(fig6, auto_open=False, output_type="div")   
            return render(request,'menu/dataanalysis2.html',{'graph_div4':graph_d3,'graph_div5':graph_d4,'graph_div6':graph_d5,'graph_div3':graph_d2,'graph_div1':graph_d,'graph_div2':graph_d1,'devices':response1.json(),'imei':imei,'pack':pack, 'location':location })
        elif(pack=="Total_Power"):
            tp=df
            plot_data = [
                go.Scatter(
                    x=tp['DeviceTimeStamp'],
                    y=tp['KWH'],
                    name='KWH'
                ),
                go.Scatter(
                    x=tp['DeviceTimeStamp'],
                    y=tp['KVARH'],
                    name='KVARH'
                )
                
            ]
            plot_layout = go.Layout(
                    title='Total Power',
                    xaxis_title='Time'
                )
            fig = go.Figure(data=plot_data, layout=plot_layout)
            plot_data = [
                go.Scatter(
                    x=tp['DeviceTimeStamp'],
                    y=tp['KW'],
                    name='KW'
                ),
                go.Scatter(
                    x=tp['DeviceTimeStamp'],
                    y=tp['KVA'],
                    name='KVA'
                )
                
            ]
            plot_layout = go.Layout(
                    title='Total Power',
                    xaxis_title='Time'
                )
            fig2 = go.Figure(data=plot_data, layout=plot_layout)
            plot_data = [
                go.Scatter(
                    x=tp['DeviceTimeStamp'],
                    y=tp['KVAR'],
                    name='KVAR'
                )
                
            ]
            plot_layout = go.Layout(
                    title='Total Power',
                    xaxis_title='Time',
                    yaxis_title='KVAR'
                )

            fig3 = go.Figure(data=plot_data, layout=plot_layout)
            plot_data = [
                go.Scatter(
                    x=tp['DeviceTimeStamp'],
                    y=tp['MPD'],
                    name='MPD'
                ),
                go.Scatter(
                    x=tp['DeviceTimeStamp'],
                    y=tp['MKVAD'],
                    name='MKVAD'
                )
                
            ]
            plot_layout = go.Layout(
                    title='Total Power',
                    xaxis_title='Time'
                )
            fig4 = go.Figure(data=plot_data, layout=plot_layout)
            graph_d = pyoff.plot(fig, auto_open=False, output_type="div")
            graph_d1 = pyoff.plot(fig2, auto_open=False, output_type="div")
            graph_d2 = pyoff.plot(fig3, auto_open=False, output_type="div")   
            graph_d3 = pyoff.plot(fig4, auto_open=False, output_type="div")
            return render(request,'menu/dataanalysis2.html',{'graph_div4':graph_d3,'graph_div3':graph_d2,'graph_div1':graph_d,'graph_div2':graph_d1,'devices':response1.json(),'imei':imei,'pack':pack, 'location':location })
        elif(pack=="overview"):
            ov=df
            ov.head()
            ov.isnull().sum()
            plot_data = [
                go.Scatter(
                    x=ov['DeviceTimeStamp'],
                    y=ov['OTI'],
                    name='OTI'
                ),
                go.Scatter(
                    x=ov['DeviceTimeStamp'],
                    y=ov['ATI'],
                    name='ATI'
                )
                
            ]
            plot_layout = go.Layout(
                    title='Overview',
                    xaxis_title='Time'
                )
            fig = go.Figure(data=plot_data, layout=plot_layout)
            plot_data = [
                go.Scatter(
                    x=ov['DeviceTimeStamp'],
                    y=ov['OLI'],
                    name='OLI'
                )
                
            ]
            plot_layout = go.Layout(
                    title='Overview',
                    xaxis_title='Time',
                    yaxis_title='OLI'
                )
            fig2 = go.Figure(data=plot_data, layout=plot_layout)
            plot_data = [
                go.Scatter(
                    x=ov['DeviceTimeStamp'],
                    y=ov['OTI_A'],
                    name='OTI_A'
                ),
                go.Scatter(
                    x=ov['DeviceTimeStamp'],
                    y=ov['OTI_T'],
                    name='OTI_T'
                )
                
            ]
            plot_layout = go.Layout(
                    title='Overview',
                    xaxis_title='Time'
                )
            fig3 = go.Figure(data=plot_data, layout=plot_layout)
            graph_d = pyoff.plot(fig, auto_open=False, output_type="div")
            graph_d1 = pyoff.plot(fig2, auto_open=False, output_type="div")
            graph_d2 = pyoff.plot(fig3, auto_open=False, output_type="div")   
            return render(request,'menu/dataanalysis2.html',{'graph_div3':graph_d2,'graph_div1':graph_d,'graph_div2':graph_d1,'devices':response1.json(),'imei':imei,'pack':pack, 'location':location })
    else:
        location1=None
        imei1=None
        pack1=None
        if(imei!='sd' and imei!=None):
            for i in range(len(res)):
                if(res[i]['DeviceImei']==imei):
                    location1=res[i]['location']
                    break
            imei1=imei
        if(pack!='pt' and pack!=None):
            pack1=pack
        return render(request,'menu/dataanalysis2.html',{'imei':imei1,'pack':pack1,'location':location1,'devices':response1.json()})
        
    
