#!/usr/bin/env python
# coding: utf-8

# # Prepare bottle data for ML model
# Created by Ivan Lima on Wed Jan 11 2023 20:35:51 -0500

import pandas as pd
import xarray as xr
import numpy as np
import os, datetime, glob, warnings
from ccsm_utils import find_stn, find_stn_idx, find_closest_pt, extract_loc
from cesm_utils import da2ma
print('Last updated on {}'.format(datetime.datetime.now().ctime()))

pd.options.display.max_columns = 50
warnings.filterwarnings('ignore')

# ## Load bottle data

df_bottle = pd.read_csv('data/CODAP_combined.csv', parse_dates=['Date'], na_values=['<undefined>',-999])

# add seasons
df_bottle.loc[df_bottle.Date.dt.month.isin([12, 1, 2]),'Season'] = 'winter'
df_bottle.loc[df_bottle.Date.dt.month.isin([3, 4, 5]),'Season'] = 'spring'
df_bottle.loc[df_bottle.Date.dt.month.isin([6, 7, 8]),'Season'] = 'summer'
df_bottle.loc[df_bottle.Date.dt.month.isin([9, 10, 11]),'Season'] = 'fall'

# select variables
cols = ['Accession', 'EXPOCODE', 'Cruise_ID', 'Observation_type', 'Station_ID', 'Cast_number',
        'Niskin_ID', 'Sample_ID', 'Date', 'Latitude', 'Longitude', 'Season', 'CTDPRES',
        'Depth', 'CTDTEMP_ITS90', 'CTDTEMP_flag', 'recommended_Salinity_PSS78',
        'recommended_Salinity_flag', 'recommended_Oxygen', 'recommended_Oxygen_flag',
        'DIC', 'DIC_flag', 'TALK', 'TALK_flag']

# rename some variables
col_new_names = {'CTDPRES': 'Pressure',
                 'CTDTEMP_ITS90': 'Temperature',
                 'CTDTEMP_flag': 'Temperature_flag',
                 'recommended_Salinity_PSS78': 'Salinity',
                 'recommended_Salinity_flag': 'Salinity_flag',
                 'recommended_Oxygen': 'Oxygen',
                 'recommended_Oxygen_flag': 'Oxygen_flag'}
df_bottle = df_bottle[cols].rename(columns=col_new_names)

# ## Clean data 

# Data flags:
#  - Flag = 2: good data (keep)
#  - Flag = 6: average of lab reps (keep). 
#  - Flag = 3: questionable data (remove)
#  - Flag = 9: no measurement (remove)
#  - Flag = NaN: no flag given

df_bottle_dic = df_bottle.loc[df_bottle.DIC_flag.isin([2, 6])].drop(['DIC_flag'], axis=1)
df_bottle_dic = df_bottle_dic.loc[df_bottle_dic.Temperature_flag.isin([2, 6])].drop('Temperature_flag', axis=1)
df_bottle_dic = df_bottle_dic.loc[df_bottle_dic.Salinity_flag.isin([2, 6])].drop('Salinity_flag', axis=1)

df_bottle_ta = df_bottle.loc[df_bottle.TALK_flag.isin([2, 6])].drop(['TALK_flag'], axis=1)
df_bottle_ta = df_bottle_ta.loc[df_bottle_ta.Temperature_flag.isin([2, 6])].drop('Temperature_flag', axis=1)
df_bottle_ta = df_bottle_ta.loc[df_bottle_ta.Salinity_flag.isin([2, 6])].drop('Salinity_flag', axis=1)


# ## Add atmospheric pCO2 data (annual mean) 

# load at pCO2 data
df_atm_pco2 = pd.read_csv('data/co2_annmean_mlo.csv', skiprows=59)
df_atm_pco2 = df_atm_pco2.rename(columns={'mean':'pCO2_atm'}).drop('unc', axis=1)

df_bottle_dic['year'] = df_bottle_dic.Date.dt.year
df_bottle_dic = pd.merge(df_bottle_dic, df_atm_pco2, on='year').drop('year', axis=1)

df_bottle_ta['year'] = df_bottle_ta.Date.dt.year
df_bottle_ta = pd.merge(df_bottle_ta, df_atm_pco2, on='year').drop('year', axis=1)


# ## Add satellite data

# extract satellite data at observation dates & locations

def extract_satellite_data(df_in):
    ssh_dir = '/bali/data/ilima/Satellite_Data/AVISO/daily/'
    sst_dir = '/bali/data/ilima/Satellite_Data/SST/NOAA_OI/'
    # sst_hr_dir = '/bali/data/ilima/Satellite_Data/SST/PO.DAAC/'
    # chl_dir = '/bali/data/ilima/Satellite_Data/Ocean_Color/Chl/daily/'
    # kd490_dir = '/bali/data/ilima/Satellite_Data/Ocean_Color/KD490/daily/'
    sst_hr_dir = '/home/ivan/Data/Postproc/Satellite_Data/PO.DAAC/'
    chl_dir = '/home/ivan/Data/Postproc/Satellite_Data/CHL/'
    kd490_dir = '/home/ivan/Data/Postproc/Satellite_Data/KD490/'
    
    df_obs = df_in.copy()
    for i in df_obs.index:
        year, month, day = df_obs.loc[i,'Date'].year, df_obs.loc[i,'Date'].month, df_obs.loc[i,'Date'].day
        doy = df_obs.loc[i,'Date'].day_of_year
        
        print('record {:4d}/{}'.format(i, df_obs.index.max()))

        # extract AVISO SSH data
        ssh_file = glob.glob(ssh_dir + '{}/{:02}/dt_global_allsat_phy_l4_{}{:02}{:02}_????????.nc'.format(year,month,year,month,day))
        if ssh_file:
            with xr.open_dataset(ssh_file[0]) as ds:
                lon_sat, lat_sat = np.meshgrid(ds.longitude, ds.latitude)
                lon_obs, lat_obs = df_obs.loc[i,['Longitude','Latitude']]
                lon_obs = lon_obs + 360.
                for var in ['adt','sla']:
                    df_obs.loc[i,var.upper()] = extract_loc(lon_obs, lat_obs, lon_sat, lat_sat, da2ma(ds[var]))
        else:
            print('SSH i={} ({}-{:02}-{:02})'.format(i, year, month, day), end=', ')

        # extract SST (0.25 x 0.25 degree) data
        sst_file = glob.glob(sst_dir + '{}/{:03d}/{}*AVHRR_OI*.nc'.format(year,doy,year))
        if sst_file:
            with xr.open_dataset(sst_file[0]) as ds:
                lon_sat, lat_sat = np.meshgrid(ds.lon, ds.lat)
                lon_obs, lat_obs = df_obs.loc[i,['Longitude','Latitude']]
                data = da2ma(ds['analysed_sst'].squeeze() - 273.15) # Kelvin -> Celsius
                df_obs.loc[i,'SST'] = extract_loc(lon_obs, lat_obs, lon_sat, lat_sat, data)
        else:
            print('SST1 i={} ({}-{:02}-{:02})'.format(i, year, month, day), end=', ')

        # extract high res SST (0.01 x 0.01 degree) data
        # sst_hr_file = sst_hr_dir + '{}/{:03d}/{}{:02}{:02}090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc'.format(year,doy,year,month,day)
        sst_hr_file = sst_hr_dir + 'subset_{}{:02}{:02}090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc'.format(year,month,day)
        if os.path.isfile(sst_hr_file):
            with xr.open_dataset(sst_hr_file) as ds:
                lon_sat, lat_sat = np.meshgrid(ds.lon, ds.lat)
                lon_obs, lat_obs = df_obs.loc[i,['Longitude','Latitude']]
                data = da2ma(ds['analysed_sst'].squeeze() - 273.15) # Kelvin -> Celsius
                df_obs.loc[i,'SST_hires'] = extract_loc(lon_obs, lat_obs, lon_sat, lat_sat, data)
        else:
            print('SST2 i={} ({}-{:02}-{:02})'.format(i, year, month, day), end=', ')

        # extract surface Chl (~4.64 Km resolution)
        # chl_file = chl_dir + '{}/{:02}/{}{:02}{:02}_d-ACRI-L4-CHL-MULTI_4KM-GLO-REP.nc'.format(year,month,year,month,day)
        chl_file = chl_dir + 'subset_{}{:02}{:02}_d-ACRI-L4-CHL-MULTI_4KM-GLO-REP.nc'.format(year,month,day)
        if os.path.isfile(chl_file):
            with xr.open_dataset(chl_file) as ds:
                lon_sat, lat_sat = np.meshgrid(ds.lon, ds.lat)
                lon_obs, lat_obs = df_obs.loc[i,['Longitude','Latitude']]
                data = da2ma(ds['CHL'].squeeze())
                df_obs.loc[i,'Chl'] = extract_loc(lon_obs, lat_obs, lon_sat, lat_sat, data)
        else:
            print('Chl i={} ({}-{:02}-{:02})'.format(i, year, month, day), end=', ')

        # extract surface KD490 (~4.64 Km resolution)
        # kd490_file = kd490_dir + '{}/{:02}/{}{:02}{:02}_d-ACRI-L4-KD490-MULTI_4KM-GLO-REP.nc'.format(year,month,year,month,day)
        kd490_file = kd490_dir + '/subset_{}{:02}{:02}_d-ACRI-L4-KD490-MULTI_4KM-GLO-REP.nc'.format(year,month,day)
        if os.path.isfile(kd490_file):
            with xr.open_dataset(kd490_file) as ds:
                lon_sat, lat_sat = np.meshgrid(ds.lon, ds.lat)
                lon_obs, lat_obs = df_obs.loc[i,['Longitude','Latitude']]
                data = da2ma(ds['KD490'].squeeze())
                df_obs.loc[i,'KD490'] = extract_loc(lon_obs, lat_obs, lon_sat, lat_sat, data)
        else:
            print('KD490 i={} ({}-{:02}-{:02})'.format(i, year, month, day), end=' | ')

    return df_obs

# ## Save cleaned data to CSV file

df_bottle_dic = extract_satellite_data(df_bottle_dic)
df_bottle_dic.to_csv('data/bottle_data_DIC_prepared.csv')

df_bottle_ta = extract_satellite_data(df_bottle_ta)
df_bottle_ta.to_csv('data/bottle_data_TA_prepared.csv')
