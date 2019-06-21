import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as de
import os

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go



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



st = pd.to_datetime(f'{min(y2u)}0101',format='%Y%m%d')
ed = pd.to_datetime(f'{max(y2u)}1231',format='%Y%m%d')
dates = []
i = st
while i <= ed:
    dates.append(i)
    i += de.timedelta(days=1)
print(dates[-1])
print(len(dates))

denom = pd.DataFrame( [[x,y] for x in dates for y in ds.loca.unique()], columns=['dt','loca'])
print(denom.shape)
print(type(denom.dt[0]))
print(ds.shape)
ds = denom.merge(ds, on=['dt','loca'], how='outer', validate='1:1')
print(ds.shape)

ds['day'] = (ds['dt'] - pd.to_datetime(ds['dt'].dt.year.astype(str)+'0101', format='%Y%m%d')).dt.days
ds['y'] = ds['dt'].dt.year
ds['m'] = ds['dt'].dt.month

locas = ['서울전체'] + list(ds.loca.unique())
seoul_daily = ds[['dt','pm10','pm25']].groupby(['dt']).mean().reset_index()
seoul_daily['loca'] = '서울전체'
seoul_daily.head()
daily_ds = seoul_daily.append(ds[['dt','loca','pm10','pm25']]).reset_index()
daily_ds['day_value'] = ((daily_ds['dt'] - st) / np.timedelta64(1,'D')).astype(int)

daily_dict = {}
for r in locas:
    daily_dict[r] = daily_ds[daily_ds['loca']==r]



# begin dashing
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

o_list = []
for y in y2u:
    o_list.append({'label':f'{str(y)}', 'value':y})
print(o_list)

loca_list = []
for l in locas:
    loca_list.append({'label':l, 'value':l})

n_days = (ds.dt.max() - ds.dt.min()).days
i = st
c = 0
day_slider_lab = {}
while i <= ed:
    if ((i.day == 1) & (i.month == 1)) | (i == ed):
        day_slider_lab[c] = f'{i.year}/{i.month}/{i.day}'
    i += de.timedelta(days=1)
    c += 1



app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),
    html.Div(children='''Dash: A web application framework for Python.'''),

    html.Label('which variable to check dropdown'),
    dcc.Dropdown(
        id = 'pm_bubble',
        options=[
            {'label':'PM10', 'value':'pm10'},
            {'label':'PM2.5','value':'pm25'}
        ],
        value='pm25'
    ),

    html.Label('base var dropdown'),
        dcc.Dropdown(
            id = 'base_var_dropdown',
            options=loca_list,
            value='서울전체'
        ),
    html.Label('comp var dropdown'),
        dcc.Dropdown(
            id = 'comp_var_dropdown',
            options=loca_list,
            value='강남구'
            ),

    dcc.Graph( id='daily_plot' ),

    html.Label('time slider'),
        dcc.RangeSlider(
            id = 'day_range',
            min = 0,
            max = n_days,
            marks = day_slider_lab,
            value = [0,n_days]
        )
])



# @app.callback(
#     Output('monthly_plot', 'figure'),
#     [Input('pm_bubble', 'value'),
#      Input('base_var_dropdown', 'value'),
#      Input('comp_var_dropdown', 'value'),
#      Input('date_range', 'value') ])
# def update_graph(   pm_bubble,
#                     year_bubble,
#                     measure_dropdown,
#                     base_var_dropdown,
#                     comp_var_dropdown,
#                     date_range):
@app.callback(
    Output('daily_plot', 'figure'),
    [Input('pm_bubble', 'value'),
     Input('base_var_dropdown', 'value'),
     Input      ('day_range', 'value') ])
def update_graph(   pm_bubble,
                    base_var_dropdown,
                        day_range):

    ds2u = daily_dict[base_var_dropdown]

    # Create a trace
    trace = go.Scatter(
        x = ds2u.loc[(ds2u['day_value']>=min(day_range)) & (ds2u['day_value']<=max(day_range)), 'dt'],
        y = ds2u.loc[(ds2u['day_value']>=min(day_range)) & (ds2u['day_value']<=max(day_range)), pm_bubble]
    )

    fig = dict(data = [trace])
    return(fig)





if __name__ == '__main__':
    app.run_server(debug=True)
