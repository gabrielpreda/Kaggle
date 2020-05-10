import os
import re
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd

covid_repo_path = "..\\..\\COVID-19"
db_source = os.path.join(covid_repo_path, "csse_covid_19_data\\csse_covid_19_daily_reports")
print(f"list of files: {len(os.listdir(db_source))}")

data_df = pd.DataFrame()
for file in tqdm(os.listdir(db_source)):
    try:
        crt_date, crt_ext = crt_file = file.split(".")
        if(crt_ext == "csv"):
            crt_date_df = pd.read_csv(os.path.join(db_source, file))
            crt_date_df['date_str'] = crt_date
            crt_date_df['Date'] = crt_date_df['date_str'].apply(lambda x: datetime.strptime(x, "%m-%d-%Y"))
            
            if(crt_date >=  '03-22-2020'):
                #break
                # FIPS, Admin2, Province_State, Country_Region, Last_Update,Lat, Long, Confirmed, Deaths, Recovered, Active, Combined_Key
                #print(f"Conversion start: {crt_date_df.columns}")
                crt_date_df['Province/State'] = crt_date_df['Province_State']
                crt_date_df['Country/Region'] = crt_date_df['Country_Region']
                crt_date_df['Latitude'] = crt_date_df['Lat']
                crt_date_df['Longitude'] = crt_date_df['Long_']
                crt_date_df = crt_date_df[['Country/Region', 'Province/State', 'Latitude', 'Longitude', 'Confirmed', 'Recovered', 'Deaths', 'Date', 'date_str']]
                
            data_df = data_df.append(crt_date_df)
    except ex as Exception:
        print(ex)
        pass
    
print(f"Data: rows: {data_df.shape[0]}, cols: {data_df.shape[1]}")
print(f"Days: {data_df.date_str.nunique()} ({data_df.date_str.min()} : {data_df.date_str.max()})")
print(f"Country/Region: {data_df['Country/Region'].nunique()}")
print(f"Province/State: {data_df['Province/State'].nunique()}")
print(f"Confirmed all  (Province/State): {sum(data_df.groupby(['Province/State'])['Confirmed'].max())}")
print(f"Confirmed all (Country/Region): {sum(data_df.groupby(['Country/Region'])['Confirmed'].max())}")
print(f"Recovered all (Province/State): {sum(data_df.loc[~data_df.Recovered.isna()].groupby(['Province/State'])['Recovered'].max())}")
print(f"Recovered all (Country/Region): {sum(data_df.loc[~data_df.Recovered.isna()].groupby(['Country/Region'])['Recovered'].max())}")      
print(f"Deaths all (Province/State): {sum(data_df.loc[~data_df.Deaths.isna()].groupby(['Province/State'])['Deaths'].max())}")
print(f"Deaths all (Country/Region): {sum(data_df.loc[~data_df.Deaths.isna()].groupby(['Country/Region'])['Deaths'].max())}")
      
province_state = data_df['Province/State'].unique()

for ps in province_state:

    data_df.loc[(data_df['Province/State']==ps) & (data_df['Latitude'].isna()), 'Latitude'] =\
                data_df.loc[(~data_df['Latitude'].isna()) & \
                            (data_df['Province/State']==ps), 'Latitude'].median()
    
    data_df.loc[(data_df['Province/State']==ps) & (data_df['Longitude'].isna()), 'Longitude'] =\
            data_df.loc[(~data_df['Longitude'].isna()) & \
                        (data_df['Province/State']==ps), 'Longitude'].median() 
      
country_region = data_df['Country/Region'].unique()

for cr in country_region:

    data_df.loc[(data_df['Country/Region']==cr) & (data_df['Latitude'].isna()), 'Latitude'] =\
                data_df.loc[(~data_df['Latitude'].isna()) & \
                            (data_df['Country/Region']==cr), 'Latitude'].median()
    
    data_df.loc[(data_df['Country/Region']==cr) & (data_df['Longitude'].isna()), 'Longitude'] =\
            data_df.loc[(~data_df['Longitude'].isna()) & \
                        (data_df['Country/Region']==cr), 'Longitude'].median() 

data_df.loc[data_df['Country/Region']==' Azerbaijan', 'Country/Region'] = 'Azerbaijan'
data_df.loc[data_df['Country/Region']=='Czechia', 'Country/Region'] = 'Czech Republic'
data_df.loc[data_df['Country/Region']=="Cote d'Ivoire", 'Country/Region'] = 'Ivory Coast'
data_df.loc[data_df['Country/Region']=="Dominica", 'Country/Region'] = 'Dominican Republic'
data_df.loc[data_df['Country/Region']=='Iran (Islamic Republic of)', 'Country/Region'] = 'Iran'
data_df.loc[data_df['Country/Region']=='Hong Kong SAR', 'Country/Region'] = 'Hong Kong'
data_df.loc[data_df['Country/Region']=='Holy See', 'Country/Region'] = 'Vatican City'
data_df.loc[data_df['Country/Region']=='Macao SAR', 'Country/Region'] = 'Macau'
data_df.loc[data_df['Country/Region']=='Mainland China', 'Country/Region'] = 'China'
data_df.loc[data_df['Country/Region']=='Niger', 'Country/Region'] = 'Nigeria'
data_df.loc[data_df['Country/Region']=='occupied Palestinian territory', 'Country/Region'] = 'West Bank and Gaza'
data_df.loc[data_df['Country/Region']=='Palestine', 'Country/Region'] = 'West Bank and Gaza'
data_df.loc[data_df['Country/Region']=='Republic of Ireland', 'Country/Region'] = 'Ireland'
data_df.loc[data_df['Country/Region']=='Korea, South', 'Country/Region'] = 'South Korea'
data_df.loc[data_df['Country/Region']=='Republic of Ireland', 'Country/Region'] = 'Ireland'
data_df.loc[data_df['Country/Region']=='Republic of Korea', 'Country/Region'] = 'South Korea'
data_df.loc[data_df['Country/Region']=='Republic of Moldova', 'Country/Region'] = 'Moldova'
data_df.loc[data_df['Country/Region']=='Republic of the Congo', 'Country/Region'] = 'Congo (Brazzaville)'
data_df.loc[data_df['Country/Region']=='Russian Federation', 'Country/Region'] = 'Russia'
data_df.loc[data_df['Country/Region']=='Taiwan*', 'Country/Region'] = 'Taiwan'
data_df.loc[data_df['Country/Region']=='The Gambia', 'Country/Region'] = 'Gambia'
data_df.loc[data_df['Country/Region']=='Gambia, The', 'Country/Region'] = 'Gambia'
data_df.loc[data_df['Country/Region']=='UK', 'Country/Region'] = 'United Kingdom'
data_df.loc[data_df['Country/Region']=='Viet Nam', 'Country/Region'] = 'Vietnam'

data_df.loc[data_df['Country/Region']=='Ivory Coast', 'Longitude'] = 5.54
data_df.loc[data_df['Country/Region']=='Ivory Coast', 'Latitude'] = 7.54
data_df.loc[data_df['Country/Region']=='North Ireland', 'Longitude'] = 6.4923
data_df.loc[data_df['Country/Region']=='North Ireland', 'Latitude'] = 54.7877
      
data_df = data_df[['Country/Region', 'Province/State', 'Latitude', 'Longitude', 'Confirmed', 'Recovered', 'Deaths', 'Date']]

      
def fix_data_for_country_date(country, date, confirmed, recovered, deaths):
    data_df.loc[(data_df['Country/Region']==country) & (data_df['Date']==date), 'Confirmed'] = confirmed
    data_df.loc[(data_df['Country/Region']==country) & (data_df['Date']==date), 'Recovered'] = recovered
    data_df.loc[(data_df['Country/Region']==country) & (data_df['Date']==date), 'Deaths'] = deaths

def fix_data_for_france_date(country, date, confirmed, recovered, deaths):
    data_df.loc[(data_df['Country/Region']==country) & (data_df['Province/State']==country) & (data_df['Date']==date), 'Confirmed'] = confirmed
    data_df.loc[(data_df['Country/Region']==country) & (data_df['Province/State']==country)  & (data_df['Date']==date), 'Recovered'] = recovered
    data_df.loc[(data_df['Country/Region']==country) & (data_df['Province/State']==country)  & (data_df['Date']==date), 'Deaths'] = deaths

      
fix_data_for_country_date('Italy', '2020-03-12', 15113, 1258, 1016)
fix_data_for_country_date('Spain', '2020-03-12', 2950, 183, 84)
fix_data_for_france_date('France', '2020-03-12', 2896, 12, 61)
fix_data_for_country_date('Switzerland', '2020-03-12', 815, 4, 4)
fix_data_for_country_date('Germany', '2020-03-12', 2369, 25, 5)
fix_data_for_country_date('Netherlands', '2020-03-12', 614, 25, 5)
      
us_data_df = data_df[(data_df['Country/Region'].isin(['US'])) &\
        ~(data_df['Province/State'].isna())]
# number of US states & exceptions
print(us_data_df[~(us_data_df['Province/State'].str.contains(","))]['Province/State'].nunique())
us_data_df[~(us_data_df['Province/State'].str.contains(","))]['Province/State'].unique()
      
data_df.to_csv("covid-19-all.csv", index=False)      