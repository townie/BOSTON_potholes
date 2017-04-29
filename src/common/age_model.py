
# coding: utf-8

# In[1]:

import pandas as pd
import pandas_profiling as pdp
import matplotlib
import matplotlib.pyplot as plt


get_ipython().magic('matplotlib inline')


# In[2]:

# !pip install pandas_profiling


# In[71]:

df=pd.read_csv("../../data/311__Service_Requests.csv", parse_dates=True, encoding='UTF-8', infer_datetime_format=True)


# In[72]:

df.head(1)


# In[73]:

potdf = df.loc[df['TYPE'].str.contains('Pot')]




# In[74]:

potdf['age'] =  pd.to_datetime(potdf['CLOSED_DT']) - pd.to_datetime(potdf['OPEN_DT'])
# potdf['expected_age'] =  pd.to_datetime(potdf['TARGET_DT']) - pd.to_datetime(potdf['OPEN_DT'])
# potdf['performance'] = potdf['expected_age'] - potdf['age'] 


# In[75]:

#remove closed claims
potdf['age'] =potdf['age'] #.dropna()
# potdf['expected_age'] = potdf['expected_age'].fillna(0)
# potdf['performance'] = potdf['performance'].fillna(0)


# In[76]:

# closedpotdf
potdf


# In[114]:

closedpotdf = potdf['age']
potdf['age_in_days'] = closedpotdf.apply(lambda x: x.days if isinstance(x, pd.Timedelta) else 0)
potdf['age_in_days'] = potdf['age_in_days'].dropna()# potdf['expected_age']

potdf['age_in_hours'] = closedpotdf.apply(lambda x: int(x.to_pytimedelta().total_seconds() /60 /60 ) if isinstance(x, pd.Timedelta) else 0)
potdf['age_in_hours'] = potdf['age_in_hours'] #.dropna()# potdf['expected_age']


# In[108]:

# s = closedpotdf[137]
# pys = s.to_pytimedelta().total_seconds() /60 /60 
# pys.total_seconds() /60 /60 


# In[139]:

# FOR DATAROBOT
drdf = potdf
drdf= drdf.drop('TARGET_DT', 1,errors='ignore')
drdf= drdf.drop('expected_age', 1,errors='ignore')
# drdf= drdf.drop('CLOSED_DT', 1,errors='ignore')

drdf= drdf.drop('SubmittedPhoto', 1,errors='ignore')
drdf= drdf.drop('ClosedPhoto', 1,errors='ignore') # cleaned_df['ClosedPhoto'][147]
drdf= drdf.drop('age_in_days', 1,errors='ignore') # cleaned_df['ClosedPhoto'][147]
drdf= drdf.drop('land_usage', 1,errors='ignore')
drdf= drdf.drop('Property_Type', 1,errors='ignore')
drdf= drdf.drop('Property_ID', 1,errors='ignore')
# drdf= drdf.drop('age', 1,errors='ignore')
drdf= drdf.drop('performance', 1,errors='ignore')
drdf= drdf.drop('LATITUDE', 1,errors='ignore')
drdf= drdf.drop('LONGITUDE', 1,errors='ignore')

drdf= drdf.drop('Geocoded_Location', 1,errors='ignore')

non_employee_source = ['Citizens Connect App','Constituent Call', 'Self Service', 'Twitter']
drdf = drdf.loc[(potdf['CASE_STATUS'] == 'Closed' ) &(potdf['Source'].isin(non_employee_source)) ]
drdf = drdf.drop('CLOSURE_REASON',1,errors='ignore')

drdf= drdf.drop('CASE_STATUS', 1, errors='ignore') #we dont need this anymore


# In[ ]:

drdf.head(10)
drdf.to_csv('dr_potholes_hours.csv')


# In[136]:

drold = drdf.loc[(drdf['age_in_hours'] >= 24 ) ]
drold.shape


# In[ ]:

drdrAgeLabel = drdf
drdrAgeLabel = drdf['age_in_hours'] >= 48

labels[137]
drdrAgeLabel.to_csv('dr_potholes_48hrs.csv')


# In[161]:

f = drdf[(drdf['age_in_hours'] >= 2400 ) ]
f


# In[132]:

potdf= potdf.drop('TARGET_DT', 1,errors='ignore')
potdf= potdf.drop('expected_age', 1,errors='ignore')
potdf= potdf.drop('CLOSED_DT', 1,errors='ignore')
potdf= potdf.drop('CLOSURE_REASON', 1,errors='ignore')
potdf= potdf.drop('Property_Type', 1,errors='ignore')
potdf= potdf.drop('Property_ID', 1,errors='ignore')
potdf= potdf.drop('LATITUDE', 1,errors='ignore')
potdf= potdf.drop('LONGITUDE', 1,errors='ignore')
potdf= potdf.drop('land_usage', 1,errors='ignore')
potdf= potdf.drop('OnTime_Status', 1,errors='ignore')
potdf= potdf.drop('Geocoded_Location', 1,errors='ignore')
potdf= potdf.drop('SubmittedPhoto', 1,errors='ignore')
potdf= potdf.drop('neighborhood_services_district', 1,errors='ignore')
potdf= potdf.drop('performance', 1,errors='ignore')
potdf= potdf.drop('Location', 1,errors='ignore') # TODO figure out better geo data usage

potdf= potdf.drop('LOCATION_STREET_NAME', 1,errors='ignore') # TODO figure out better geo data usage
potdf= potdf.drop('ClosedPhoto', 1,errors='ignore') # cleaned_df['ClosedPhoto'][147]



non_employee_source = ['Citizens Connect App','Constituent Call', 'Self Service', 'Twitter']
cleaned_df = potdf.loc[(potdf['CASE_STATUS'] == 'Closed' ) &(potdf['Source'].isin(non_employee_source)) ]


cleaned_df= cleaned_df.drop('CASE_STATUS', 1, errors='ignore') #we dont need this anymore

print(cleaned_df.columns)
cleaned_df.head()


# In[133]:

import pandas_profiling as pdp
cleaned_report = pdp.ProfileReport(cleaned_df)
cleaned_report.to_file(outputfile="closed_potholes.html")
cleaned_report


# In[ ]:

cleaned_df= cleaned_df.drop('SUBJECT', 1,errors='ignore')  #looks liked we missed SUBJECT
cleaned_df.columns


# In[ ]:

# Drop examples where age_in_days is greater than 174 (99% of all pots are closed in 174 days)
cut_off =174
cleaned_df = cleaned_df[cleaned_df["age_in_days"] < cut_off]
cleaned_df


# In[ ]:




# # Lets Prep our data

# In[ ]:

all_cols = ['CASE_ENQUIRY_ID', 'OPEN_DT', 'CASE_TITLE', 'REASON', 'TYPE', 'QUEUE',
       'Department', 'fire_district', 'pwd_district', 'city_council_district',
       'police_district', 'neighborhood', 'ward', 'precinct',
       'LOCATION_ZIPCODE', 'Source', 'age', 'age_in_days']

label = 'age_in_days'

metadata = ['CASE_ENQUIRY_ID',]

categorials = [ 'OPEN_DT', 'CASE_TITLE', 'REASON', 'TYPE', 'QUEUE',
       'Department', 'pwd_district', 
       'police_district', 'neighborhood', 'ward', 
       'LOCATION_ZIPCODE', 'Source']

numericals = ['precinct','city_council_district','fire_district',  'age']


# In[ ]:

pre = cleaned_df.shape[0] 
post = cleaned_df.dropna().shape[0]
diff = pre-post
print("Pre: {}, Post: {}, Diff: {}".format(pre, post , diff))


# In[ ]:

# Seems good enough to just skip this, if we need more data... come back?!!@?#

dropna_for_cleaned_df = True

if dropna_for_cleaned_df:
    cleaned_df = cleaned_df.dropna()

cleaned_df.shape


# In[ ]:

# !pip install sklearn


# In[ ]:

try:
    from sklearn.model_selection import train_test_split
except:
    from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    cleaned_df[categorials + numericals], cleaned_df[label], test_size=0.33, random_state=42)


# In[ ]:




# In[23]:

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
# enc = OneHotEncodingTransformer()
# enc.fit_transform(X_train['neighborhood'])  
# pd.get_dummies(X_train['CASE_TITLE']).shape
# case = X_train['neighborhood']
# case.__class__
# vec.fit_transform(case.to_dict())
# # X_train.to_numeric()
# case.to_dict()


# In[22]:

X_train.to_dict( orient = 'records' )
# vec.fit_transform({"case" :case})


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



