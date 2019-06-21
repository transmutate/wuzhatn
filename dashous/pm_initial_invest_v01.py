#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as de
import os

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', -1)

import seaborn as sns
sns.set(context='paper', style='darkgrid', rc={'figure.facecolor':'white'}, font_scale=1.2)


# In[4]:


win_path = 'C:/Users/noname/Desktop/wuzhatn'
mac_path = '/Users/noname/documents/github/wuzhatn'
main = win_path if os.path.exists(win_path) else mac_path
raw = main + '/raw'

years = range(2011,2019,1)
y2u = range(2014,2019,1)


# In[80]:


a = pd.read_excel(f'{raw}/일별평균대기오염도_2018.xlsx', usecols=[0,1,6,7])
t = a.columns[3]
print(len(t))
unit = t[-4:-1]
print(unit)


# In[92]:


all_ds = {}
for y in years:
    all_ds[y] = pd.read_excel(f'{raw}/일별평균대기오염도_{str(y)}.xlsx', usecols=[0,1,6,7])
    all_ds[y].dropna(axis=1, how='all', inplace=True)


# In[6]:


ds = pd.DataFrame()
for y in y2u:
    ds = ds.append(all_ds[y])

ds.columns = ['dt','loca','pm10','pm25']
ds['dt'] = pd.to_datetime(ds['dt'], format='%Y%m%d')

ds = ds[ds['loca'].str[-1]=='구']
ds.reset_index(inplace=True, drop=True)

print(ds.loca.unique())
print(ds.dt.min())
print(ds.dt.max())
print(ds.shape)
print(ds.loca.value_counts())
print(ds.loca.value_counts().shape)
print(type(ds.dt[1]))
ds.head()


# In[7]:


locas = ds.loca.unique()

st = pd.to_datetime(f'{min(y2u)}0101',format='%Y%m%d')
ed = pd.to_datetime(f'{max(y2u)}1231',format='%Y%m%d')
dates = []
i = st
while i <= ed:
    dates.append(i)
    i += de.timedelta(days=1)
print(dates[-1])
print(len(dates))
print(locas)

denom = pd.DataFrame( [[x,y] for x in dates for y in locas], columns=['dt','loca'])
print(denom.shape)
print(type(denom.dt[0]))
print(ds.shape)
ds = denom.merge(ds, on=['dt','loca'], how='outer', validate='1:1')
print(ds.shape)

ds['day'] = (ds['dt'] - pd.to_datetime(ds['dt'].dt.year.astype(str)+'0101', format='%Y%m%d')).dt.days
ds['y'] = ds['dt'].dt.year
ds['m'] = ds['dt'].dt.month

ds.head()


# In[8]:


print(ds[(ds['dt'].dt.year==2018) & (ds['loca']=='강동구') & (ds['pm10'].isna()==True)].shape )
ds[(ds['dt'].dt.year==2018) & (ds['loca']=='강동구') & (ds['pm10'].isna()==True)].head()
ds.head()


# In[62]:


# collapse at month-district level
mgb = ds[['loca','pm10','pm25','y','m']].groupby(['loca','y','m']).agg(['mean','min','max','median'])
mgb.columns = pd.Index(i[0]+'_'+i[1] for i in mgb.columns.tolist())
mgb.reset_index(inplace=True)

# now get a month-level one based on all districts
agb = ds[['pm10','pm25','y','m']].groupby(['y','m']).agg(['mean','min','max','median'])
agb.columns= pd.Index(i[0]+'_'+i[1] for i in agb.columns.tolist())
agb.reset_index(inplace=True)
agb['loca'] = 'all'

gb = agb.append(mgb).reset_index(drop=True)

gb.head()


# In[63]:


meas_dict = {}
for reg in gb.loca.unique():
    meas_dict[reg] = {}
    for v in ['pm10','pm25']:
        meas_dict[reg][v] = {}
        for meas in ['mean','median','min','max']:
            meas_dict[reg][v][meas] = {}
            for m in range(1,13):
                t = []
                for y in y2u:
                    t.append(float(gb.loc[(gb['loca']==reg) & (gb['m']==m)                                           & (gb['y']==y), v+'_'+meas]))
                meas_dict[reg][v][meas][m] = t
# print(meas_dict)


# In[34]:


import plotly
plotly.__version__
plotly.tools.set_credentials_file(username='transmutate', api_key='84x96fZlmp68dwfftTHY')


# In[91]:


def plot_monthly_trend_over_the_years(measure,reg,pm):
    from plotly import tools
    import plotly.plotly as py
    import plotly.graph_objs as go

    all_dict = {}
    trace_dict = {}
    for i in range(1,13):
        dstring = '1999' + str(i) + '01' if len(str(i))==2 else '19990' + str(i) + '01'
        dval = pd.to_datetime(dstring, format='%Y%m%d')
        mo_lab = dval.strftime("%b")

        if i == 1:
            name_fill_base = '서울전체'
            name_fill_comp = reg
            show_leg_stat = True
        else:
            name_fill_base = ''
            name_fill_comp = ''
            show_leg_stat = False

        all_dict[i] = go.Scatter(x = list(y2u),
                                 y = meas_dict['all'][pm][measure][i],
                                 hoverinfo = 'y',
                                 marker = dict(color = 'red'),
                                 name = name_fill_base,
                                 showlegend = show_leg_stat)

        trace_dict[i] = go.Scatter(x = list(y2u),
                                   y = meas_dict[reg][pm][measure][i],
                                   hoverinfo = 'y',
                                   marker = dict(color = 'blue'),
                                   name = name_fill_comp,
                                   showlegend = show_leg_stat)

    fig = tools.make_subplots(rows=1, cols=12, shared_yaxes=True,                               subplot_titles=('Jan','Feb','Mar','Apr','May','Jun',
                                              'Jul','Aug','Sep','Oct','Nov','Dec'))
    for i in range(1,13):
        fig.append_trace(all_dict[i], 1, i)
        fig.append_trace(trace_dict[i], 1, i)

    meas_txt_dict = {'min':'Minimum','max':'Maximum','mean':'Average','median':'Median'}
    pm_txt_dict = {'pm10':'PM10','pm25':'PM2.5'}
    title_txt = f'{meas_txt_dict[measure]} {pm_txt_dict[pm]} Level: Seoul vs {reg}'

    fig['layout'].update(height=400, width=1200, title=title_txt)
    fig['layout']['yaxis1'].update(title=f'{pm_txt_dict[pm]} Level in {unit}')

    return(py.iplot(fig, filename='multiple-subplots-shared-yaxes'))



print('****************************')
print('****************************')
print('****************************')
plot_monthly_trend_over_the_years('median','동작구','pm25')
