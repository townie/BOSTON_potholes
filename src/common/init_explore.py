
# coding: utf-8

# # City of Boston 311 Data Exploration
# 
# Digging into what data is on 311, Service RequestsCity Services
# 
# https://data.cityofboston.gov/City-Services/311-Service-Requests/awu8-dc52

# In[42]:

# Initialize Python Modules 
import pandas as pd
import pandas_profiling as pdp
import matplotlib
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')


# # Load Data
# 
# https://data.cityofboston.gov/City-Services/311-Service-Requests/awu8-dc52

# In[2]:

df=pd.read_csv("../../data/311__Service_Requests.csv", parse_dates=True, encoding='UTF-8', infer_datetime_format=True)


# In[3]:

df.head()


# In[4]:

# Show all columns on intial DF
df.columns


# In[43]:

# df.describe()


# In[44]:

# Heavy weight profiling. 
# TODO: Find HUGE columns
# noisey_cols = ['LATITUDE', 'OPEN_DT', 'TARGET_DT','CLOSED_DT', 'CASE_ENQUIRY_ID', 'Location', 'LOCATION_STREET_NAME', 'Property_ID', 'LATITUDE','LONGITUDE', 'LOCATION_ZIPCODE', ]
# pdp.ProfileReport(df[df.columns.difference(noisey_cols)])
important_cols = ['OnTime_Status',
       'CASE_STATUS', 'CLOSURE_REASON', 'CASE_TITLE', 'SUBJECT', 'REASON',
       'TYPE', 'QUEUE', 'Department', 'SubmittedPhoto', 'ClosedPhoto'
       , 'fire_district', 'pwd_district', 'city_council_district',
       'police_district', 'neighborhood', 'neighborhood_services_district',
       'ward', 'precinct', 'land_usage', 'Property_Type', 'Source', 'Geocoded_Location']
# profile = pdp.ProfileReport(df[important_cols])


# In[45]:

# profile.to_file(outputfile="full_profile.html")
# profile


# In[46]:

potdf = df.loc[df['TYPE'].str.contains('Pot')]
potdf.head()


# In[11]:

pothole_profile = pdp.ProfileReport(potdf[important_cols])


# In[12]:

pothole_profile.to_file(outputfile="pothole_profile.html")

pothole_profile


# # Setup Data Partitions
# 
# ### Citizen Reported DF
# 
# ### City Worker Reported DF

# In[13]:

# Citizen Reported DF

non_employee_source = ['Citizens Connect App','Constituent Call', 'Self Service', 'Twitter']
citizen_df = potdf.loc[potdf['Source'].isin(non_employee_source) ]

citizen_df.head(1)['Source']


# In[14]:

# City Worker Reported DF

criterion = lambda row: row['Source'] not in non_employee_source
worker_df = potdf[potdf.apply(criterion, axis=1)]

worker_df.head(1)['Source']


# # Potholes by Neigborhood

# In[15]:

# helper methods
import collections

def graph_pots(dd, t=None, yl=None, xl=None):
    l = range(len(dd.keys()))
    plt.bar(l, dd.values(), align='center')
    plt.xticks(l, dd.keys(), rotation='vertical')
    plt.ylabel(yl)
    plt.xlabel(xl)
    plt.title(t)
    plt.show()
    
def graph_col(col_df, col_name):
    return graph_pots(
        collections.Counter(col_df[col_name].dropna())
        )

def potholes_by(hole_df, place, df_name):
    return graph_pots(
        collections.Counter(hole_df[place].dropna()),
        yl='Closed Potholes',
        xl='Places',
        t="Potholes by {} Reported By {}".format(place, df_name)
        )


# In[16]:

df.index


# In[17]:


potholes_by(potdf, 'neighborhood', 'All')


# In[18]:

potholes_by(citizen_df, 'neighborhood', 'Citizens')
potholes_by(worker_df, 'neighborhood', 'Workers')


# # Potholes by City Council District

# In[19]:

potholes_by(potdf, 'city_council_district', 'All')


# In[20]:

potholes_by(citizen_df, 'city_council_district', 'Citizens')
potholes_by(worker_df, 'city_council_district', 'Workers')


# # Age of Potholes
# 
# 'OPEN_DT', 'TARGET_DT', 'CLOSED_DT'
# 
# `age =  DateClosed('CLOSED_DT') - DateCreated('OPEN_DT')`
# 
# `expected_age = TargetDate('TARGET_DT') - DateCreate('OPEN_DT') `
# 
# `performance = expected_age - age`

# In[21]:

potdf['age'] =  pd.to_datetime(potdf['CLOSED_DT']) - pd.to_datetime(potdf['OPEN_DT'])
potdf['expected_age'] =  pd.to_datetime(potdf['TARGET_DT']) - pd.to_datetime(potdf['OPEN_DT'])
# Datepotdf['performance'] = potdf['expected_age'] - potdf['age'] 


# In[22]:

# understanding datetimes in pythons
print("Timedelta for a good date vs nan date ")

good_date = potdf.age[51]
nan_date = potdf.age[973808]

print("isinstance(good_date, pd.Timedelta): " + str(isinstance(good_date, pd.Timedelta)))
print("good.__class__ == pd.tslib.Timedelta: " + str(good_date.__class__ == pd.tslib.Timedelta))
try:
    good_date.days
    print("good_date.days")
except:
    good_date.day
    print("good_date.day")
    
print()

print("isinstance(nan_date, pd.Timedelta): " + str(isinstance(nan_date, pd.Timedelta)))
print("nan_date.__class__ == pd.tslib.Timedelta: " + str(nan_date.__class__ == pd.tslib.Timedelta))
try:
    nan_date.days
    print("nan_date.days")
except:
    nan_date.day
    print("nan_date.day")


# In[23]:

closedpotdf = potdf['age'].dropna()
potdf['age_in_days'] = closedpotdf.apply(lambda x: x.days if isinstance(x, pd.Timedelta) else 0)
# potdf['age_in_days'].dropna()


# In[24]:

# import collections

# # df = df[df.line_race != 0]

# dd = collections.Counter(potdf['neighborhood'].dropna())
# age_dd = collections.Counter(potdf['age_in_days'].dropna())
# age_dd
# # # plt.bar(dd.values(), dd.keys())
# # l = range(len(60)
# plt.hist(x=potdf['age_in_days'], data=potdf['neighborhood'])
# # # plt.xticks(l, age_dd.keys(), rotation='vertical')
# # plt.show()


# In[25]:

potdf['age_in_days'].describe()


# In[26]:

# p = potdf['age_in_days'].plot.hist()
# plt.show()


# In[ ]:




# In[27]:

from collections import defaultdict
age_map = defaultdict(int) 

for age in potdf['age_in_days'].iteritems():
    age_map[age[1]] += 1


# In[28]:

age_map


# In[31]:

non_employee_source = ['Citizens Connect App','Constituent Call', 'Self Service', 'Twitter']
citizens_df = potdf.loc[(potdf['CASE_STATUS'] == 'Closed' ) &(potdf['Source'].isin(non_employee_source)) ]
citizens_df.head()


# In[48]:


citizens_age = defaultdict(int) 

for age in citizens_df['age_in_days'].iteritems():
    citizens_age[age[1]] += 1
    
citizens_age

top_n = 40
age_keys = sorted(citizens_age.keys())

print("first {} days:\n".format(top_n))
for i in age_keys[:top_n]:
    print(i, citizens_age[i])
    
print()

bottom_n = 5
print("last {} days:\n".format(bottom_n))
for i in age_keys[len(age_keys) - bottom_n:]:
    print(i, citizens_age[i])
    


# # Picking arbitray cut off
# 
# Essentially, we have a long tail. We should try different cut offs. But figure out what those % are.}

# ## cumulative_days

# In[33]:

cumulative_days =  defaultdict(int) 
seen = []
for k in age_keys:
    seen.append(k)
    for older in seen:
        cumulative_days[k] += citizens_age[older]

for i in range(45):
    print(i, cumulative_days[i])


# In[50]:

# Day by Cumulative Number of Closed Potholes

x1 = []
y1 = []
for date in cumulative_days:
    x1.append(date)
    y1.append(cumulative_days[date])
plt.ylabel('closed potholes')
plt.xlabel('days_old')
plt.title('Cumulative Number of Closed Potholes by Day')
plt.bar(x1, y1)
plt.show()


# In[35]:

n = citizens_df.shape[0]
print("N examples: "+str(n))


# In[36]:

percent_by_day =  defaultdict(int) 
seen = []
for k in cumulative_days:
    percent_by_day[k] = cumulative_days[k]/n

percent_by_day  

for i in range(25): # 95% percentile
    print(i, percent_by_day[i])


# In[59]:

# compute_percent_changes_points
prev = 0.0

change_pts = []
change_chart = {}
for indx in percent_by_day:
    current = (percent_by_day[indx] * 100)
    if (current - prev >= 1) and (current != 0):
        change_pts.append(indx)
        print("CHANGE @ {} current:{} prev:{}".format(indx,current, prev))
        prev = current
        
        
percent_by_day[31]    


# In[74]:

age_by_percent_x = [percent_by_day[pt] for pt in change_pts]
# percent_by_day.keys()
# change_pts
# age_by_percent_xp
plt.plot(age_by_percent_x,change_ptse
plt.ylabel('closed potholes')
plt.xlabel('days_old')
plt.title('Cut off of {} '.format(cut_off))
plt.show()


# In[38]:

cut_off = 24 # 99

for i in range(170, 175): # 95% percentile
    print(i, percent_by_day[i])


# In[39]:

cut_off_dates = [25, 73]
for cut_off in cut_off_dates:
    x = []
    y = []
    for date in range(cut_off):
        x.append(citizens_age[date])
        y.append(date)
    plt.bar( y, x, log=False)
    plt.ylabel('closed potholes')
    plt.xlabel('days_old')
    plt.title('Cut off of {} '.format(cut_off))
    plt.show()


# In[40]:

citizens_age


# In[41]:

citizens_df


# In[ ]:




# In[ ]:



