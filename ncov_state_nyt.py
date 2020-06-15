#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 11:55:34 2020
n-cov tracker from public sources
@author: cdnozzle
"""
import pandas as pd
import streamlit as st
import cufflinks as cf
from sklearn.linear_model import LinearRegression
import numpy as np
from urllib.request import urlopen
import json
import plotly.express as px


@st.cache(allow_output_mutation=True)
def get_data_link(link):
    data = pd.read_csv(link)
    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)

    return data,counties

@st.cache()
def calc_scatter(select,confirmed,dma_roll):
    full_scat=pd.DataFrame()
    reg_val  =pd.DataFrame()
    for i in select:
       scat=pd.concat([confirmed[i],confirmed[i].diff().rolling(dma_roll).mean()],axis=1)
       scat.columns = ['Level','Change']
       scat.fillna(0,inplace=True)
       scat = scat[~(scat == 0).any(axis=1)]
       # Log Log regression
       X = np.log10(scat['Level']).values
       Y = np.log10(scat['Change']).values
       reg =LinearRegression().fit(X.reshape(-1,1),Y)
       reg_val.loc[i,'Coef'] = reg.coef_[0]
       scat['Country']=i
       full_scat =pd.concat([full_scat,scat])

    return reg_val,full_scat

'''
# 2019-nCoV tracker by county in the US

## Data from github repository Of [The New York Times](https://github.com/nytimes/covid-19-data)
To see New York times visualization see [link](https://www.nytimes.com/interactive/2020/us/coronavirus-us-cases.html)

'''
link = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'


data,counties = get_data_link(link)

data_last = data[data['date']==data['date'].max()]
sum_state=data_last.groupby(by='state').sum()
st.write(sum_state[['cases','deaths']])

# county by county plot
#last =data.loc[data['date']==data['date'].max(),['fips','cases']]

#fig = px.choropleth(last, geojson=counties, locations='fips', color='cases',
#                           color_continuous_scale="Viridis",
#                           range_color=(0, 12),
#                           scope="usa",
#                           labels={'cases':'Confirmed Cases'}
#                          )
#fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
#st.plotly_chart(fig)
#fig.show()




data['County, State'] = data['county'] +' ,' +data['state']
data.index = pd.MultiIndex.from_frame(data[['date','County, State']])
#st.write(data.head(n=10))
type_plot = st.sidebar.selectbox('plot type ?',('cases','deaths'))
confirmed = data[type_plot].unstack()

last_value=confirmed.iloc[-1].sort_values(ascending=False)
st.markdown('## Counties with most cases')
st.write(last_value.head(n=20))

st.markdown('## Fastest growing counties')
last_diff = confirmed.diff().rolling(7).mean().iloc[-1].sort_values(ascending=False)
st.write(last_diff.head(n=20))


select = st.multiselect('Select counties', confirmed.columns.values)
dma_roll =st.sidebar.slider('Moving avg', min_value=1,max_value=10)
st.markdown('## Time series of total cases in county' )
log_y = st.checkbox('Log scale ?')
ax = confirmed[select].iplot(asFigure=True,logy=log_y)
st.plotly_chart(ax)
st.markdown('*Data from The New York Times, based on reports from state and local health agencies.*')

st.markdown('## Daily New cases')
ax1= confirmed[select].diff().rolling(dma_roll).mean().iplot(asFigure=True)
st.plotly_chart(ax1)
st.markdown('*Data from The New York Times, based on reports from state and local health agencies.*')

st.markdown('## County growth rates vs total cases')


reg_val, full_scat = calc_scatter(select,confirmed[select],dma_roll)
ax2 =full_scat.iplot(asFigure=True,x='Level',y='Change',mode='markers',logx=True,logy=True,categories='Country',xTitle='Total Cases',
                      yTitle='New Cases')
st.plotly_chart(ax2)
st.markdown('*Data from The New York Times, based on reports from state and local health agencies.*')

st.markdown('## Regression Log (Change) vs Log( Level)')

st.write(reg_val)
