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

@st.cache()
def get_data_link(link):
    data = pd.read_csv(link)
    return data
'''
# 2019-nCoV tracker by county in the US

## Data from github repository Of [New York times](https://github.com/nytimes/covid-19-data)
'''
link = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'

data = get_data_link(link)
data['County, State'] = data['county'] +' ,' +data['state']
data.index = pd.MultiIndex.from_frame(data[['date','County, State']])
#st.write(data.head(n=10))
confirmed = data['cases'].unstack()

last_value=confirmed.iloc[-1].sort_values(ascending=False)
st.markdown('## Counties with most cases')
st.write(last_value.head(n=20))


select = st.sidebar.multiselect('Select counties', confirmed.columns.values)

st.markdown('## Time series of total cases in county' )
ax = confirmed[select].iplot(asFigure=True,logy=True)
st.plotly_chart(ax)

st.markdown('## Daily New cases')
ax1= confirmed[select].diff().iplot(asFigure=True)
st.plotly_chart(ax1)

st.markdown('## County growth rates vs total cases')
# #country_sel = st.selectbox('Select country',confirmed.columns.values)

full_scat=pd.DataFrame()
reg_val  =pd.DataFrame()
for i in select:
    scat=pd.concat([confirmed[i],confirmed[i].diff()],axis=1)
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


#st.write(scat)
ax2 =full_scat.iplot(asFigure=True,x='Level',y='Change',mode='markers',logx=True,logy=True,categories='Country',xTitle='Total Cases',
                      yTitle='New Cases')
st.plotly_chart(ax2)
st.markdown('## Regression Log (Change) vs Log( Level)')
st.write(reg_val)