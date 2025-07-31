#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install yfinance==0.2.65')


# In[2]:


get_ipython().system('pip install finta')


# In[45]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime 
from math import sqrt
import seaborn as sns
from finta import TA
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split,KFold,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
import streamlit as st
import warnings
warnings.filterwarnings('ignore')


# In[4]:


import yfinance as yf


# In[5]:


yf.enable_debug_mode()


# In[6]:


Coca_Cola=yf.Ticker("KO")


# In[7]:


Coca_Cola_info=Coca_Cola.info


# In[8]:


Coca_Cola_info


# In[9]:


info_df=pd.json_normalize(Coca_Cola_info)
info_df


# ##  Load the dataset into a dataframe

# In[10]:


KO_df=Coca_Cola.history(period="max")
KO_df.head()


# In[12]:


KO_df.tail()


# In[11]:


KO_df1=KO_df.copy()


# In[12]:


KO_df.reset_index(inplace=True)
KO_df.head()


# In[17]:


KO_df.info()


# In[13]:


#df['Date']=pd.to_datetime(df['Date'],errors='coerce')


# In[14]:


#df['Year']=df['Date'].dt.year


# In[19]:


KO_df.isnull().sum()


# In[13]:


Q1_close=KO_df['Close'].quantile(0.25)
Q3_close=KO_df['Close'].quantile(0.75)
IQR_close=Q3_close-Q1_close
max_close=Q3_close + (1.5 * IQR_close)
close_df=KO_df[KO_df['Close']>max_close]
close_df



# In[14]:


plt.figure(figsize=(12,6))
plt.plot(close_df['Close'])
plt.title('Trend of Close')
plt.xlabel('Date')
plt.ylabel('Close Price in USD greater than 3 standard deviation which is considered as threshold')
plt.show()


# The stock shows a bullish trend from 2019-07-03 onwards. 

# In[15]:


KO_df.plot(subplots=True,figsize=(10,12),grid=True)
plt.title('Coca Cola Stock Attributes')
plt.show()


# In[16]:


plt.figure(figsize=(12,6))
plt.plot(KO_df['Close'])
plt.title('Close price history for stock history of Coca Cola')
plt.ylabel('Close Price in USD($) for Coca Cola',fontsize=18)
plt.show()


# In[17]:


plt.figure(figsize=(12,6))
plt.plot(KO_df['Open'])
plt.title('Open price history for stock history of Coca Cola')
plt.ylabel('Open Price in USD($) for Coca Cola',fontsize=18)
plt.show()


# In[18]:


plt.figure(figsize=(12,6))
plt.plot(KO_df['Volume'])
plt.title('Distribution of volume(total amount of trading activity) each day through years')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.show()


# In[19]:


plt.figure(figsize=(12,6))
plt.plot(KO_df['Dividends'])
plt.title('Distribution of dividends')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.show()


# In[31]:


#df['Stock Splits'].unique()


# In[38]:


#df['Date']=pd.to_datetime(df['Date'],format='mixed',utc=True)


# In[39]:


#df.info()


# In[20]:


KO_df['Year']=KO_df['Date'].dt.year


# In[33]:


KO_df.isnull().sum()


# In[21]:


df_grp=KO_df.groupby('Year')['Stock Splits'].count().to_frame().reset_index()
df_grp.columns=['Year','Stock Split Counts']
df_grp


# In[22]:


plt.figure(figsize=(12,6))
sns.barplot(y='Stock Split Counts',x='Year',data=df_grp,palette=sns.color_palette('Set2'))
plt.xlabel('Year')
plt.ylabel('Stock Split')
plt.xticks(rotation=90)
plt.show()


# The stock splits follow a constant trend. 

# In[46]:


#Coca_Cola=yf.Ticker("KO")


# In[47]:


#Coca_Cola_info=Coca_Cola.info


# In[48]:


#Coca_Cola_info


# In[49]:


#info_df=pd.json_normalize(Coca_Cola_info)
#info_df


# In[50]:


#Coca_Cola_data=Coca_Cola.history(period="max")
#Coca_Cola_data.head()


# In[23]:


def fin_indicators():
    
    df1=pd.DataFrame()
    #Moving averages
    df1['MA_20']=KO_df['Close'].rolling(window=20).mean()
    df1['MA_50']=KO_df['Close'].rolling(window=50).mean()
    df1['MA_252']=KO_df['Close'].rolling(window=252).mean()
    df1['RSI']=TA.RSI(KO_df)
    df1['Average_True_Range']=TA.ATR(KO_df)
    df1['On_Balance_Volume']=TA.OBV(KO_df)
    df1['Volume_Flow_indicator']=TA.VFI(KO_df)
    df1['Volume_Weighted_Average_Price']=TA.VWAP(KO_df)
    df1['Exponential_Moving_Average_50']=TA.EMA(KO_df,50)
    df1['ADX']=TA.ADX(KO_df)
    df1['William%R']=TA.WILLIAMS(KO_df)
    df1['MFI']=TA.MFI(KO_df)
    df1['MOM']=TA.MOM(KO_df)
    df1[['macd','macd_signal']]=TA.MACD(KO_df)
    df1[['Buy_Pressure','Sell_Pressure']]=TA.BASP(KO_df,period=40)
    df1[['Bull_power','Bear_power']]=TA.BASP(KO_df,period=40)



    #Add Daily returns
    df1['Daily_Return']=KO_df['Close'].pct_change()
    df1['Price_Up_or_Down']=np.where(df1['Daily_Return']<0,-1,1)

    #Add volatility
    df1['Volatility']=df1['Daily_Return'].rolling(window=20).std()

    return df1


# In[24]:


df1=fin_indicators()


# In[25]:


df1.head()


# In[26]:


df1.fillna(0,inplace=True)


# In[27]:


df1.head()


# In[28]:


df1.isnull().sum()


# In[29]:


df=pd.concat([KO_df,df1],axis=1)
df.head()


# In[30]:


df.isnull().sum()


# In[44]:


df.info()


# In[31]:


plt.figure(figsize=(15,12))
df[['Close','MA_50','MA_252']].plot()
plt.xlabel('Year')
plt.ylabel('Moving averages')
plt.show()


# In[32]:


plt.figure(figsize=(15,12))
df[['Daily_Return']].hist(bins=50,sharex=True,figsize=(12,8))
plt.ylabel('Close Price')
plt.xlabel('Percentage Change')
plt.show()


# In[33]:


min_period=75
vol=df1['Daily_Return'].rolling(min_period).std()*np.sqrt(min_period)
vol.plot(figsize=(12,6))


# The stock is highly volatile.

# In[34]:


pd.plotting.scatter_matrix(df[['Daily_Return']],diagonal='kde',alpha=0.1,figsize=(12,6))
plt.xlabel('Percentage change of close price')
plt.ylabel('Close Price')
plt.show()


# In[35]:


cor=df.corr()
cor


# In[36]:


plt.figure(figsize=(12,6))
sns.heatmap(KO_df.corr(),annot=True,cmap='coolwarm')
plt.show()


# In[37]:


df_corr=df.corr()['Close'].drop('Close')
df_corr=df_corr.sort_values()
df_corr.plot(kind='barh',figsize=(12,6))


# ## Model Building

# In[38]:


X=df[['Open','Low','High','Volume','Dividends','Stock Splits']]
y=df['Close']


# In[39]:


x_scale=StandardScaler().fit_transform(X)


# In[40]:


X_train,X_test,Y_train,Y_test=train_test_split(x_scale,y,test_size=0.2,random_state=42)


# In[41]:


rf=RandomForestRegressor(n_estimators=100,random_state=42)


# In[42]:


rf.fit(X_train,Y_train)


# In[43]:


y_pred=rf.predict(X_test)


# In[44]:


print('Mean Squared Error is ',mean_squared_error(Y_test,y_pred))
print('R2 Score is',r2_score(Y_test,y_pred))


# In[107]:


Close_Price_Pred=pd.DataFrame()
Close_Price_Pred['Close Price']=y_pred


# In[108]:


Close_Price_Pred.to_csv("D://Unified mentor internship projects//Coca_Cola_Stock_Price.csv",index=False,header=True)


# ## Deploy the system

# In[46]:


st.title('Coca Cola Stock Live and Updated')
st.subheader('DataFrame')
st.dataframe(KO_df)

st.title('Line Chart')
st.line_chart(KO_df[['Close','MA_20','MA_50']])



# In[ ]:




