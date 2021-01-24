import pandas as pd
import numpy as np

# Using pandas read CSV as data frame
df = pd.read_csv('xyz.csv')

#help us find how long the customer had been onboard
# in our data we columns referencing each month from column x to y
# here we use time frame 2 years for our model 
# which is why you see range from 1 to 25 (24 months for 2 years)
df_r = pd.DataFrame(np.multiply(df.iloc[:,x:y].values,np.array(range(1,25))))

#append ['CUS_ID','JOIN_DATE'] from df to df_r
df_r[['CUS_ID','JOIN_DATE']]=df[['CUS_ID','JOIN_DATE']]

# Convert ['JOIN_DATE'] data type from Object to datetime64
df_r['JOIN_DATE']=pd.to_datetime(df_r['JOIN_DATE'])

# "Transpose" data using melt this will create N * 24 rows
# We find the max number of ActiveLength which will be used for our recency calculation
df_r2 = pd.DataFrame(pd.melt(df_r,
                     id_vars=['CUS_ID','JOIN_DATE'],var_name='isActiveMonth',value_name='ActiveLength').\
             groupby('CUS_ID')['ActiveLength'].max()).reset_index()

# Append column ['ActiveLength'] from df_r2 to new df_r3 which contains ['CUS_ID'],['JOIN_DATE'] and ['ActiveLength']
df_r3 = df_r[['CUS_ID','JOIN_DATE']].merge(df_r2,on='CUS_ID')

# Calculate the how long customer has joined (in month)
df_r3['month_joined'] = df_r3['JOIN_DATE'].apply(lambda x:(2020-x.year)*12+(9-x.month))
# Calculate recency by calculating the difference between joined length and active length
df_r3['recency'] = df_r3['month_joined'] - df_r3['ActiveLength']

df_frequency = pd.DataFrame(df.iloc[:,0])
# In our data we have columns indicating whether customer made purchase that month using 0 or 1
# We calculate frequency by summing up values under those columns (x through y)
df_frequency['frequency']=df.loc[:,"x":"y"].sum(axis =1)

df_monetary = pd.DataFrame(df.iloc[:,0])
# We have a column total_spend indicating total amount customer spend last 2 years for our case
df_monetary['monetary']=df['total_spend']

df_test = df_monetary[df_monetary['monetary'] < 0]

#Remove duplicated columns
df_RFM_Detail = df_RFM.loc[:,~df_RFM.columns.duplicated()]

#Remove details columns
df_RFM_Clean = df_RFM_Detail.drop(columns = ["JOIN_DATE","ActiveLength","month_joined"],axis=1)
df_RFM_Clean.columns = ['CustomerID','Recency', 'Monetary', 'Frequency']

#Step 2: To automate the segmentation we will use 80% quantile for Recency and Monetary
# We will use the 80% quantile for each feature
quantiles = df_RFM_Clean.quantile(q=[0.8])
print(quantiles)
df_RFM_Clean['R']=np.where(df_RFM_Clean['Recency']<=int(quantiles.Recency.values), 2, 1)
df_RFM_Clean['F']=np.where(df_RFM_Clean['Frequency']>=int(quantiles.Frequency.values), 2, 1)
df_RFM_Clean['M']=np.where(df_RFM_Clean['Monetary']>=int(quantiles.Monetary.values), 2, 1)

#Step 3: Calculate RFM score and sort customers
# To do the 2 x 2 matrix we will only use Recency & Monetary
df_RFM_Clean['RFMScore'] = df_RFM_Clean.M.map(str)+df_RFM_Clean.F.map(str)+df_RFM_Clean.R.map(str)
#df_RFM = df_RFM.reset_index()
df_RFM_SUM = df_RFM_Clean.groupby('RFMScore').agg({'CustomerID': lambda y: len(y.unique()),
                                        'Frequency': lambda y: round(y.mean(),0),
                                        'Recency': lambda y: round(y.mean(),0),
                                        'R': lambda y: round(y.mean(),0),
                                        'M': lambda y: round(y.mean(),0),
                                        'F': lambda y: round(y.mean(),0),
                                        'Monetary': lambda y: round(y.mean(),0)})
df_RFM_SUM = df_RFM_SUM.sort_values('RFMScore', ascending=False)

# Write results into CSV file
df_RFM_SUM.to_csv('xyz.csv')
