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
import numpy as np
from sklearn.linear_model import LinearRegression

@st.cache(allow_output_mutation=True)
def get_data_link(link):
    data = pd.read_csv(link)
    return data

@st.cache()
def calc_scatter(select,confirmed):
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
       
    return reg_val,full_scat

'''
# 2019-nCoV tracker by Country 

## Data from github repository [link](https://github.com/datasets/covid-19)
'''
link = 'https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv'

data = get_data_link(link)
data.index=pd.MultiIndex.from_frame(data[['Date','Country']])
#st.write(data.head(n=10))
confirmed=data['Confirmed'].unstack()

select = st.sidebar.multiselect('Select Countries', confirmed.columns.values)

st.markdown('## Time series of total cases' )
ax = confirmed[select].iplot(asFigure=True,logy=True)
st.plotly_chart(ax)

st.markdown('## Daily New cases')
ax1= confirmed[select].diff().iplot(asFigure=True)
st.plotly_chart(ax1)

st.markdown('## Country growth rates vs total cases')
#country_sel = st.selectbox('Select country',confirmed.columns.values)

reg_val, full_scat = calc_scatter(select,confirmed[select])
ax2 =full_scat.iplot(asFigure=True,x='Level',y='Change',mode='markers',logx=True,logy=True,categories='Country',xTitle='Total Cases',
                     yTitle='New Cases')
st.plotly_chart(ax2)
st.markdown('## Regression Log (Change) vs Log( Level)')
st.write(reg_val)