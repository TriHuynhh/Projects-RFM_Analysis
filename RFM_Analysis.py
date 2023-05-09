# IMPORT LIBRARIES 
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt


#==============================READ IN DATA=============================================

df = pd.read_csv(r'C:\Users\Asus\Desktop\Homework and Assignments\Portfolio Projects\Retail Transaction Data\Retail_Data_Transactions.csv',
                 parse_dates=['trans_date'])


#==============================DATA MANIPULATION======================================

# Assume that the study is being done as of 01/Apr/2015. 
std_date = dt.datetime(2015,4,1)
df['hist'] = std_date - df['trans_date']
df['hist'] = df['hist']/np.timedelta64(1,'D')

# Only data from less than 2 years are considered for analysis 
df = df[df['hist'] < 2*365]

# The data will be summarized at customer level 
# by taking number of days to the latest transaction, 
# sum of all transction amount and total number of transaction.
rfm_table = df.groupby('customer_id').agg(
    {
        'hist': lambda a: a.min(),  # Recency
        'customer_id': lambda b: len(b),    # Frequency
        'tran_amount': lambda c: c.sum()    # Monetary value 
    }
)

rfm_table.rename(columns=
                 {
                     'hist':'recency',
                     'customer_id':'frequency',
                     'tran_amount':'monetary_value'
                 }, inplace=True)

quartiles = rfm_table.quantile(q=[0.25,0.50,0.75])  # Divide the dataframe into 4 different segments 

# Convert quartiles into dictionary type so that cutoffs can be picked up
quartiles = quartiles.to_dict()

# Create function to evaluate R,F,M
# x: value 
# p: recency, monetary_value, frequency
# d: quartiles dict 
def RClass(x,p,d):  # Function for Recency (the lower the better)
    if x <= d[p][0.25]: 
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]:
        return 3
    else: 
        return 4

def FMClass(x,p,d): # Function for Frequency & Monetary_value (the higher the better)
    if x <= d[p][0.25]: 
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]:
        return 2
    else: 
        return 1

# Use a different table for segmentation
rfm_seg = rfm_table
rfm_seg['R_quartile'] = rfm_seg['recency'].apply(RClass,args=('recency',quartiles))
rfm_seg['F_quartile'] = rfm_seg['frequency'].apply(FMClass,args=('frequency',quartiles))
rfm_seg['M_quartile'] = rfm_seg['monetary_value'].apply(FMClass,args=('monetary_value',quartiles))

# Combine three scores to create a single score
# For analysis it is critical to combine the scores to create a single score. 

# There are few approaches. 
# One approach is to just concatenate the scores to create a 3 digit number between 111 and 444. 
# Here the drawback is too many categories (4x4x4). 
# Also, not easy prioritise scores like 421 vs 412.

rfm_seg['RFM_class'] = rfm_seg['R_quartile'].map(str) + rfm_seg['F_quartile'].map(str) + rfm_seg['M_quartile'].map(str)     # .map(str) to convert into string for concatenate
# rfm_seg = rfm_seg.sort_values(by=['RFM_class','monetary_value'],ascending=[True,False])


# Another possibility is to combine the scores to create one score (eg. 4+1+1). 
# This will create a score between 3 and 12. 
# Here the sdvantage is that each of the scores got same importance. 
# However some scores will have many sgements as constituents (eg - 413 ad 431)

rfm_seg['Total_score'] = rfm_seg['R_quartile'] + rfm_seg['F_quartile'] + rfm_seg['M_quartile']
rfm_seg = rfm_seg.sort_values(by=['RFM_class','monetary_value'],ascending=[True,False])
# print(rfm_seg.groupby('Total_score').agg('monetary_value').mean())  # agg is used to perform another function inside a dataframe

m = rfm_seg.groupby('Total_score').agg('monetary_value').mean().plot(kind='bar',color='blue',title='monetary_value')
# plt.show()
f = rfm_seg.groupby('Total_score').agg('frequency').mean().plot(kind='bar',color='blue',title='frequency')
# plt.show()
r = rfm_seg.groupby('Total_score').agg('recency').mean().plot(kind='bar',color='blue',title='recency')
# plt.show()


# Ultimate test of RFM score is the impact on any consumer behaviour. 
# Let's check its impact on the response of customers to a promotion campaign.
response = pd.read_csv(r'C:\Users\Asus\Desktop\Homework and Assignments\Portfolio Projects\Retail Transaction Data\Retail_Data_Response.csv')

# Reset rfm_seg index
rfm_seg = rfm_seg.sort_values('customer_id',ascending=True)
rfm_seg.reset_index(inplace=True)

# Join 2 tables
rfm2 = pd.merge(rfm_seg,response,on='customer_id')

rfm2.groupby('Total_score').agg({'response':'mean'}).plot(
    kind='bar',
    color='blue',
    xlabel='Total Score',
    ylabel='Proportion of Responders'
)
plt.show()