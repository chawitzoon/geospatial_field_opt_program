''' 
read precomputed output csv for rough tuning 

the adaptation of utils/utils_crop_sim_csv.py in field-optimization-uncertainty

'''

import pandas as pd
import os
from datetime import date, datetime


def _weather_idx(string):
    filename = string.split('/')[-1]
    idx = filename.split('.')[0][-3:]
    return idx

def crop_sim_find_best(crop_sim_dir, variety, scenario_num, plant_day_search_start = 2022091, plant_range=60):

    filename_dir = os.path.join(crop_sim_dir, f"{variety}.csv")
    df_all = pd.read_csv(filename_dir)

    df_all['weather'] = df_all['weather'].apply(_weather_idx)
    df = df_all[df_all['weather']== f'{scenario_num:03}'].reset_index()

    date_first = datetime.strptime(str(plant_day_search_start), '%Y%j') # datetime object

    df['transplant_date'] = pd.to_datetime(df['transplant_date'], format='%Y-%m-%d')
    df['maturity_date'] = pd.to_datetime(df['maturity_date'], format='%Y-%m-%d')

    # will lookup dataframe, then count range as row
    date_first_index = df.index[df['transplant_date'] == date_first].tolist()[0]

    plant_day = (df['transplant_date'].iloc[date_first_index:date_first_index+plant_range] - date_first).dt.days.to_numpy()
    harvest_day = (df['maturity_date'].iloc[date_first_index:date_first_index+plant_range] - date_first).dt.days.to_numpy()
    yield_brown = df['yield_brown_rice(t/ha)'].iloc[date_first_index:date_first_index+plant_range].to_numpy()

    return yield_brown, plant_day, harvest_day
    