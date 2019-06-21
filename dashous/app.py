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
agb['loca'] = '서울전체'


gb = agb.append(mgb).reset_index(drop=True)
locas = gb.loca.unique()
print(locas)
gb.head()





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

mo_lab = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
mo_lab_dict = {}
for i,v in enumerate(mo_lab):
    mo_lab_dict[i+1] = v

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
    html.Label('year bubble dropdown'),
        dcc.Checklist(
            id = 'year_bubble',
            options = o_list,
            values = list(y2u),
            labelStyle={'display': 'inline-block', "margin-right": "20px"}
        ),

    html.Label('measure dropdown'),
        dcc.Dropdown(
            id = 'measure_dropdown',
            options=[
                {'label': 'Monthly Average', 'value': 'mean'},
                {'label': 'Monthly Median', 'value': 'median'},
                {'label': 'Monthly Minimum', 'value': 'min'},
                {'label': 'Monthly Maximum', 'value': 'max'}
            ],
            value='mean'
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

    dcc.Graph( id='monthly_plot' ),

    html.Label('month slider'),
        dcc.RangeSlider(
            id = 'mo_range',
            min=1,
            max=12,
            marks=mo_lab_dict,
            value=[1, 12]
        )
])



@app.callback(
    Output('monthly_plot', 'figure'),
    [Input('pm_bubble', 'value'),
     Input('year_bubble', 'values'),
     Input('measure_dropdown', 'value'),
     Input('base_var_dropdown', 'value'),
     Input('comp_var_dropdown', 'value'),
     Input('mo_range', 'value') ])
def update_graph(   pm_bubble,
                    year_bubble,
                    measure_dropdown,
                    base_var_dropdown,
                    comp_var_dropdown,
                    mo_range):
    year_bubble.sort()

    st = min(mo_range)
    ed = max(mo_range)
    dur = ed - st + 1

    # make a dictionary with data to be fed into the plot
    meas_dict = {}
    for reg in [base_var_dropdown, comp_var_dropdown]:
        meas_dict[reg] = {}
        for v in [pm_bubble]:
            meas_dict[reg][v] = {}
            for meas in [measure_dropdown]:
                meas_dict[reg][v][meas] = {}
                # for m in range(1,13):
                for m in range(st, ed+1):
                    t = []
                    for y in year_bubble:
                        t.append(float(gb.loc[(gb['loca']==reg) & (gb['m']==m) & (gb['y']==y), v+'_'+meas]))
                    meas_dict[reg][v][meas][m] = t

    all_dict = {}
    trace_dict = {}
    for i in range(st, ed+1):
        if i == st:
            name_fill_base = base_var_dropdown
            name_fill_comp = reg
            show_leg_stat = True
        else:
            name_fill_base = ''
            name_fill_comp = ''
            show_leg_stat = False

        all_dict[i] = go.Scatter(x = year_bubble,
                                 y = meas_dict[base_var_dropdown][pm_bubble][measure_dropdown][i],
                                 hoverinfo = 'y',
                                 marker = dict(color = 'red'),
                                 name = name_fill_base,
                                 showlegend = show_leg_stat)

        trace_dict[i] = go.Scatter(x = year_bubble,
                                   y = meas_dict[comp_var_dropdown][pm_bubble][measure_dropdown][i],
                                   hoverinfo = 'y',
                                   marker = dict(color = 'blue'),
                                   name = name_fill_comp,
                                   showlegend = show_leg_stat)

    fig = tools.make_subplots(rows=1, cols=ed-st+1, shared_yaxes=True, subplot_titles=mo_lab[st-1:ed])
    c = 1
    for i in range(st, ed+1):
        fig.append_trace(all_dict[i], 1, c)
        fig.append_trace(trace_dict[i], 1, c)
        c += 1

    meas_txt_dict = {'min':'Minimum','max':'Maximum','mean':'Average','median':'Median'}
    pm_txt_dict = {'pm10':'PM10','pm25':'PM2.5'}
    title_txt = f'{meas_txt_dict[measure_dropdown]} {pm_txt_dict[pm_bubble]} Level: {base_var_dropdown} vs {comp_var_dropdown}'

    fig['layout'].update(title=title_txt)
    fig['layout']['yaxis1'].update(title=f'{pm_txt_dict[pm_bubble]} Level in {unit}')

    for i in range(dur):
        fig['layout']['xaxis'+str(i+1)].update(tickmode='array')
        fig['layout']['xaxis'+str(i+1)].update(tickvals=year_bubble)
        fig['layout']['xaxis'+str(i+1)].update(tickangle=90)

    return(fig)





if __name__ == '__main__':
    app.run_server(debug=True)
