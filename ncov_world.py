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
# 2019-nCoV tracker by Country
## Data from github repository [link](https://github.com/datasets/covid-19)
'''
link = 'https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv'

data = get_data_link(link)
data.index=pd.MultiIndex.from_frame(data[['Date','Country']])
#st.write(data.head(n=10))
data['NetCases'] = data['Confirmed']-data['Deaths']-data['Recovered']
series=st.sidebar.selectbox('Plot type?',('Confirmed','Deaths','NetCases'),0)
confirmed=data[series].unstack()

select = st.sidebar.multiselect('Select Countries', confirmed.columns.values)


str_txt = '## Time series of total '+ series
st.markdown(str_txt)
ax = confirmed[select].iplot(asFigure=True,logy=True)
st.plotly_chart(ax)

txt_1 = '## Daily New '+ series
st.markdown(txt_1)
dma_roll =st.sidebar.slider('Moving avg', min_value=1,max_value=10)
ax1= confirmed[select].diff().rolling(dma_roll).mean().iplot(asFigure=True)
st.plotly_chart(ax1)


#country_sel = st.selectbox('Select country',confirmed.columns.values)

if (series=='NetCases'):
    st.markdown('## No Regression for Net cases')
else:
    st.markdown('## Country growth rates vs total '+series)
    reg_val, full_scat = calc_scatter(select,confirmed[select],dma_roll)
    ax2 =full_scat.iplot(asFigure=True,x='Level',y='Change',mode='markers',logx=True,logy=True,categories='Country',xTitle='Total Cases',
                     yTitle='New Cases')
    st.plotly_chart(ax2)
    st.markdown('## Regression Log (Change) vs Log( Level)')
    st.write(reg_val)
