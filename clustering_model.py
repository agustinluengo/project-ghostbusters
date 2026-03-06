#import libraries
import pandas as pd

#import data
df = pd.read_parquet(r'./9be6127a-e740-4cd0-8534-4c147d8e6a3b_short.parquet')

#checking formatting
df.dtypes

#checking null values
df.isna().sum()

# Removing 1112 values due to missing item_size, by the moment
df.dropna()

#change item_size for an integer value
df['item_size'] = df['item_size'].replace({'Small':1,'Medium':2, 'Large':3,'HeavyBulky':4})

#force formatting
col_types = {'asin':'object',
 'warehouse_id':'object', 
 'gl_product_group':'object', 
 'product_category':'object',
 'marketplace_id':'int64',
 'item_size':'int64', 
 'p_rev':'float64', 
 'pcogs':'float64', 
 'cp':'float64',
 'display_ads_amt':'float64', 
 'vfcc':'float64', 
 'sales_discount':'float64', 
 'net_ppm':'float64', 
 'units_shipped':'float64',
 'p_rev.1':'float64', 
 'pcogs.1':'float64', 
 'cp.1':'float64', 
 'display_ads_amt.1':'float64', 
 'vfcc.1':'float64',
 'sales_discount.1':'float64', 
 'net_ppm.1':'float64', 
 'units_shipped.1':'float64'}
 
df = df.astype(col_types)


#feature engineering: adding CM, NetPPM (%), asp, acu, cppu

df['CM'] = df['cp']/df['p_rev']
df['NetPPM (%)'] = df['net_ppm']/df['p_rev']
df['asp'] = df['p_rev']/df['units_shipped']
df['acu'] = df['pcogs']/df['units_shipped']
df['cppu'] = df['cp']/df['units_shipped']