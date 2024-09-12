import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from scipy import stats
from tqdm import tqdm
from models import BertEncoder
from utils import compute_hourly_stats, convert_stats_to_str
from utils import normalize_multimodal_x
from constant import PMDATA_PATH

warnings.filterwarnings('ignore')

# fitbit data
def extract_fitbit_data_hourly(path, category):
    data_sample = json.loads(open(path).read())
    data_sample = pd.DataFrame(data_sample)
    data_sample['dateTime'] = pd.to_datetime(data_sample['dateTime'])
    data_sample = data_sample.set_index(['dateTime'])
    
    if category == 'heart_rate':
        data_sample['value'] = data_sample['value'].apply(lambda x: x['bpm']).astype(float)
        data_sample = pd.DataFrame(data_sample).resample('H').mean().reset_index().rename(
            columns={'value': category},
        )
    else:
        data_sample['value'] = data_sample['value'].astype(float)
        data_sample = pd.DataFrame(data_sample).resample('H').sum().reset_index().rename(
            columns={'value': category},
        )
        
    # Extract date from dateTime and group by date to create a list of values
    data_sample['date'] = data_sample['dateTime'].dt.date
    data_sample = data_sample.groupby('date')[category].apply(list)
    return data_sample

def extract_fitbit_sleep_daily(path):
    sleep = json.loads(open(path).read())
    sleep = pd.DataFrame(sleep)
    sleep = sleep[sleep['mainSleep'] == True]
    sleep = sleep[['dateOfSleep', 'minutesAsleep', 'minutesAwake', 'timeInBed', 'efficiency']]
    sleep['dateOfSleep'] = pd.to_datetime(sleep['dateOfSleep'])
    sleep = sleep.set_index('dateOfSleep')
    sleep_score_path = os.path.join('/'.join(path.split('/')[:-1]), 'sleep_score.csv')
    sleep_scores = pd.read_csv(sleep_score_path)
    sleep_scores = sleep_scores[
        [
            'timestamp', 'overall_score', 'composition_score', 
            'revitalization_score', 'deep_sleep_in_minutes', 
            'resting_heart_rate', 'restlessness',
        ]
    ]
    sleep_scores['timestamp'] = pd.to_datetime(sleep_scores['timestamp']).dt.date
    sleep_scores = sleep_scores.set_index('timestamp')
    sleep_all = sleep.merge(sleep_scores, left_index=True, right_index=True, how='inner')
    return sleep_all

def extract_fitbit_data_daily(path, category):
    data_sample = json.loads(open(path).read())
    data_sample = pd.DataFrame(data_sample)
    data_sample['dateTime'] = pd.to_datetime(data_sample['dateTime']).dt.date
    data_sample = data_sample.set_index(['dateTime'])
    data_sample.rename(columns={'value': category}, inplace=True)
    if category == 'time_in_heart_rate_zones':
        data_sample[category] = data_sample[category].apply(
            lambda x: x['valuesInZones'] if isinstance(x, dict) and 'valuesInZones' in x else x,
        )
        expanded_values_df = data_sample[category].apply(pd.Series)
        data_sample = data_sample.drop(category, axis=1).join(expanded_values_df)
    return data_sample

def extract_fitbit(subject, category):
    path = os.path.join(PMDATA_PATH, subject, 'fitbit', f'{category}.json')
    if category == 'sleep':
        return extract_fitbit_sleep_daily(path)
    elif category in ['heart_rate', 'steps', 'calories', 'distance']:
        return extract_fitbit_data_hourly(path, category)
    else:
        return extract_fitbit_data_daily(path, category)

def get_fitbit_data():
    subjects = [f'p{f:02d}' for f in range(1, 17)]
    categories = [
        'calories', 'distance', 'heart_rate', 'steps', 'time_in_heart_rate_zones', 
        'lightly_active_minutes', 'moderately_active_minutes', 'sedentary_minutes', 
        'sleep',  'very_active_minutes',
    ]
    fitbit_data = pd.DataFrame()
    for subject in subjects:
        subject_df = None
        for category in tqdm(categories):
            try:
                data = extract_fitbit(subject, category)
                if subject_df is None:
                    subject_df = data
                else:
                    subject_df = pd.merge(
                        subject_df, data, how='inner', 
                        left_index=True, right_index=True,
                    )
            except Exception as e:
                print(f'Error extracting {category} for {subject}: {e}')
            
        subject_df = subject_df.reset_index().rename(columns={'index': 'date'})
        subject_df.insert(0, 'subject', subject)
        fitbit_data = pd.concat([fitbit_data, subject_df], axis=0)
        fitbit_data.reset_index(drop=True, inplace=True)
    return fitbit_data

# questionare prompts
PARTICIPANT_DATA = {
    'p01': {'age': 48, 'gender': 'male', 'height': 195},
    'p02': {'age': 60, 'gender': 'male', 'height': 180},
    'p03': {'age': 25, 'gender': 'male', 'height': 184},
    'p04': {'age': 26, 'gender': 'female', 'height': 163},
    'p05': {'age': 35, 'gender': 'male', 'height': 176},
    'p06': {'age': 42, 'gender': 'male', 'height': 179},
    'p07': {'age': 26, 'gender': 'male', 'height': 177},
    'p08': {'age': 27, 'gender': 'male', 'height': 186},
    'p09': {'age': 26, 'gender': 'male', 'height': 180},
    'p10': {'age': 38, 'gender': 'female', 'height': 179},
    'p11': {'age': 25, 'gender': 'female', 'height': 171},
    'p12': {'age': 27, 'gender': 'male', 'height': 178},
    'p13': {'age': 31, 'gender': 'male', 'height': 183},
    'p14': {'age': 45, 'gender': 'male', 'height': 181},
    'p15': {'age': 54, 'gender': 'male', 'height': 180},
    'p16': {'age': 23, 'gender': 'male', 'height': 182},
}
def generate_daily_prompts(df):
    prompts = []
    for _, row in df.iterrows():
        desc = []
        participant_id = row['participant_id']
        age = PARTICIPANT_DATA[participant_id]['age']
        gender = PARTICIPANT_DATA[participant_id]['gender']
        height = PARTICIPANT_DATA[participant_id]['height']
        desc.append(f'age: {age} years'.lower())
        desc.append(f'gender: {gender}'.lower())
        desc.append(f'height: {height} cm'.lower())

        meals = row['meals'] if pd.notnull(row['meals']) else None
        if meals:
            desc.append(f'meals: {meals}'.lower())
        glasses = row['glasses_of_fluid'] if pd.notnull(row['glasses_of_fluid']) else None
        if glasses:
            desc.append(f'glasses of fluid: {glasses}'.lower())
        alcohol = None if row['alcohol_consumed'] == 'No' else 'consumed'
        if alcohol:
            desc.append(f'alcohol: {alcohol}'.lower())
        if pd.notnull(row['activity_names']):
            if row['activity_names'] != ' {}':
                activity_str = f"participated in {row['activity_names']} with a perceived exertion of "
                activity_str += f"{row['perceived_exertion']} for {row['duration_min']} minutes"
                desc.append(f'activity: {activity_str}'.lower())
        
        sleep_duration = row['sleep_duration_h'] if pd.notnull(row['sleep_duration_h']) else None
        if sleep_duration:
            desc.append(f'sleep duration: {sleep_duration} hours'.lower())
        sleep_quality = row['sleep_quality'] if pd.notnull(row['sleep_quality']) else None
        if sleep_quality:
            desc.append(f'sleep quality: {sleep_quality}'.lower())
        
        fatigue = row['fatigue'] if pd.notnull(row['fatigue']) else None
        if fatigue:
            desc.append(f'fatigue: {fatigue}'.lower())
        readiness = row['readiness'] if pd.notnull(row['readiness']) else None
        if readiness:
            desc.append(f'readiness: {readiness}'.lower()) 
        soreness = row['soreness'] if pd.notnull(row['soreness']) else None
        if soreness:
            # NOTE: soreness_area is a weird integer that is not explainable
            # soreness_area = row['soreness_area'] if pd.notnull(row['soreness_area']) else None
            desc.append(f'soreness: {soreness}'.lower())
        
        had_injury = row['hadInjury']
        injuries = row['injuries']
        if had_injury:
            injury_str = f'injuries: {injuries}'.lower()
            if injury_str != 'injuries: nan':
                desc.append(injury_str)
            
        mood = row['mood'] if pd.notnull(row['mood']) else None
        if mood:
            desc.append(f'mood: {mood}'.lower())
        stress = row['stress'] if pd.notnull(row['stress']) else None
        if stress:
            desc.append(f'stress: {stress}'.lower())
            if stress > 3:
                desc.append('label: stressed'.lower())
            else:
                desc.append('label: relaxed'.lower())
        # seperate by newline
        descs = '\n'.join(desc)
        prompts.append(descs)
    return prompts

def load_data(file_path, columns):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame(columns=columns)

def process_survey(participant_id):
    # columns
    reporting_columns = [
        'date', 'timestamp', 'meals', 'weight', 'glasses_of_fluid', 'alcohol_consumed',
    ]
    srpe_columns = [
        'end_date_time', 'activity_names', 'perceived_exertion', 'duration_min',
    ]
    wellness_columns = [
        'effective_time_frame', 'fatigue', 'mood', 'readiness', 'sleep_duration_h', 
        'sleep_quality', 'soreness', 'soreness_area', 'stress',
    ]
    injury_columns = ['effective_time_frame', 'injuries']
    
    # read csv
    reporting_df_path = os.path.join(
        PMDATA_PATH, participant_id, 'googledocs', 'reporting.csv',
    )
    srpe_df_path = os.path.join(
        PMDATA_PATH, participant_id, 'pmsys', 'srpe.csv',
    )
    wellness_df_path = os.path.join(
        PMDATA_PATH, participant_id, 'pmsys', 'wellness.csv',
    )
    injury_df_path = os.path.join(
        PMDATA_PATH, participant_id, 'pmsys', 'injury.csv',
    )
    
    reporting_df = load_data(reporting_df_path, reporting_columns)
    srpe_df = load_data(srpe_df_path, srpe_columns)
    wellness_df = load_data(wellness_df_path, wellness_columns)
    injury_df = load_data(injury_df_path, injury_columns)

    reporting_df['date'] = pd.to_datetime(
        reporting_df['date'], format='%d/%m/%Y', errors='coerce',
    )
    srpe_df['end_date_time'] = pd.to_datetime(
        srpe_df['end_date_time'], errors='coerce',
    ).dt.date
    wellness_df['effective_time_frame'] = pd.to_datetime(
        wellness_df['effective_time_frame'], errors='coerce',
    ).dt.date
    injury_df['effective_time_frame'] = pd.to_datetime(
        injury_df['effective_time_frame'], errors='coerce',
    ).dt.date

    srpe_df.rename(columns={'end_date_time': 'date'}, inplace=True)
    wellness_df.rename(columns={'effective_time_frame': 'date'}, inplace=True)
    injury_df.rename(columns={'effective_time_frame': 'date'}, inplace=True)

    reporting_df['date'] = pd.to_datetime(reporting_df['date'], errors='coerce')
    srpe_df['date'] = pd.to_datetime(srpe_df['date'], errors='coerce')
    wellness_df['date'] = pd.to_datetime(wellness_df['date'], errors='coerce')
    injury_df['date'] = pd.to_datetime(injury_df['date'], errors='coerce')

    merged_df = pd.merge(reporting_df, srpe_df, on='date', how='outer')
    merged_df = pd.merge(merged_df, wellness_df, on='date', how='outer')
    merged_df = pd.merge(merged_df, injury_df, on='date', how='outer')

    merged_df['participant_id'] = participant_id

    meal_types = ['Breakfast', 'Lunch', 'Dinner', 'Evening']
    for meal in meal_types:
        merged_df[f'had{meal}'] = merged_df['meals'].apply(
            lambda x: 1 if pd.notnull(x) and meal in x else 0,
        )

    merged_df['hadInjury'] = merged_df['injuries'].apply(
        lambda x: 0 if x == '{}' else 1,
    )

    # Creating the updated feature table by including the new binary columns
    feature_columns_updated = [
        'date', 'participant_id', 'weight', 'glasses_of_fluid', 'perceived_exertion', 
        'duration_min', 'fatigue', 'mood', 'readiness', 'sleep_duration_h', 
        'sleep_quality', 'soreness', 'stress', 'hadBreakfast', 'hadLunch', 
        'hadDinner', 'hadEvening', 'hadInjury',
    ]
    feature_df_updated = merged_df[feature_columns_updated]
    
    prompts_df = merged_df[['date', 'participant_id']].copy()
    prompts_df['prompts'] = generate_daily_prompts(merged_df)

    return feature_df_updated, prompts_df

def get_survey_data():
    subjects = [f'p{f:02d}' for f in range(1, 17)]
    all_feature_df = pd.DataFrame()
    all_prompts_df = pd.DataFrame()
    for participant_id in subjects:
        feature_df, prompts_df = process_survey(participant_id)
        all_feature_df = pd.concat([all_feature_df, feature_df], axis=0)
        all_prompts_df = pd.concat([all_prompts_df, prompts_df], axis=0)
    return all_feature_df, all_prompts_df

def binarize_stress(x):
    # if NaN return -1
    if pd.isna(x):
        return -1
    # if x is 3, 4, or 5 
    if x > 3:
        return 1
    # otherwise return 0
    return 0

def compute_stats_pmdata(df_path):
    # df_path = os.path.join(PMDATA_PATH, 'processed', f'{flag}.parquet')
    stats_map = {
        'mean': np.mean,
        'std': np.std,
        'min': np.min,
        'max': np.max,
        '50': np.median,
        'kurtosis': stats.kurtosis,
        'skew': stats.skew,
        'iqr': stats.iqr,
    }
    df_all = pd.read_parquet(df_path)
    df_name = os.path.basename(df_path)
    df_name = df_name.split('.')[0]
    df_dir = os.path.dirname(df_path)

    fitbit_cols = [
        'steps',
        'heart_rate',
        'calories',
        'distance',
    ]
    fitbit = df_all[fitbit_cols]
    fitbit = fitbit.apply(lambda x: np.stack(x), axis=1)
    fitbit = np.stack(fitbit)
    fitbit = np.nan_to_num(fitbit)

    # compute stats for 24 hours in fitbit data
    fitbit_stats = {}
    for k, v in stats_map.items():
        fitbit_stats[k] = v(fitbit, axis=2)

    percentile_25 = np.percentile(fitbit, 25, axis=2)
    percentile_75 = np.percentile(fitbit, 75, axis=2)
    rms = np.sqrt(np.mean(fitbit ** 2, axis=2))

    fitbit_stats['25'] = percentile_25
    fitbit_stats['75'] = percentile_75
    fitbit_stats['rms'] = rms

    fitbit_stats_field_df = pd.DataFrame()
    for key, value in fitbit_stats.items():
        df = pd.DataFrame(value, columns=[col + '_' + key for col in fitbit_cols])
        fitbit_stats_field_df = pd.concat([fitbit_stats_field_df, df], axis=1)

    tabular_cols = [
        # location
        'BELOW_DEFAULT_ZONE_1', 'IN_DEFAULT_ZONE_1', 'IN_DEFAULT_ZONE_3', 'IN_DEFAULT_ZONE_2', 
        # active
        'lightly_active_minutes', 'moderately_active_minutes', 'sedentary_minutes', 'very_active_minutes', 
        # heart
        'efficiency', 'resting_heart_rate',
        # scores
        'overall_score', 'composition_score', 'revitalization_score',
        # location
        'IN_CUSTOM_ZONE', 'BELOW_CUSTOM_ZONE', 'ABOVE_CUSTOM_ZONE', 
        # demographics
        'age', 'weight', 'height', 'gender',
        # drink
        'glasses_of_fluid', 
        # activity
        'perceived_exertion', 'duration_min', 
        # mood
        'mood', 'restlessness',
        # sleep
        'minutesAsleep', 'minutesAwake', 'timeInBed',
        'deep_sleep_in_minutes', 'sleep_duration_h', 'sleep_quality', 
        # body
        'soreness', 'fatigue', 'readiness', 'hadInjury',
        # eat
        'hadBreakfast', 'hadLunch', 'hadDinner', 'hadEvening',
    ]
    key_df = df_all[tabular_cols]
    key_df = key_df.reset_index()
    # drop the index column
    key_df = key_df.drop(columns='index')
    key_df = key_df.fillna(0)
    fitbit_stats_field_df = pd.concat([fitbit_stats_field_df, key_df], axis=1)

    label_df = df_all['stress_label'].reset_index()
    fitbit_stats_field_df = pd.concat([fitbit_stats_field_df, label_df], axis=1)
    # drop the index column
    fitbit_stats_field_df = fitbit_stats_field_df.drop(columns='index')
    
    # all dtypes are float64
    fitbit_stats_field_df = fitbit_stats_field_df.astype(np.float64)
    
    # to parquet
    fitbit_stats_field_df.to_parquet(
        os.path.join(df_dir, f'{df_name}_stats.parquet'), index=False,
    )
    return fitbit_stats_field_df

def update_prompts_column(row):
    description = []
    steps = row['steps']
    bpm = row['heart_rate']
    calories = row['calories']
    distance = row['distance']
    steps_stats = compute_hourly_stats(steps)  # return a dictionary keyed by stat type
    bpm_stats = compute_hourly_stats(bpm)
    calories_stats = compute_hourly_stats(calories)
    distance_stats = compute_hourly_stats(distance)
    steps_stats_str = f'steps: {convert_stats_to_str(steps_stats)}'
    bpm_stats_str = f'heart rate: {convert_stats_to_str(bpm_stats)}'
    calories_stats_str = f'calories: {convert_stats_to_str(calories_stats)}'
    distance_stats_str = f'distance: {convert_stats_to_str(distance_stats)}'
    description.append(steps_stats_str)
    description.append(bpm_stats_str)
    description.append(calories_stats_str)
    description.append(distance_stats_str)
    description = '\n'.join(description)
    description = description + '\n'
    
    prompts = row['prompts']
    prompts = description + prompts
    prompts = prompts.split('\n')
    # exclude attributes
    prompts = [d for d in prompts if d.split(':')[0].strip() not in ['stress', 'mood']]
    # join desc
    prompts = '\n'.join(prompts)

    return prompts

def main():
    save_dir = os.path.join(PMDATA_PATH, 'user_split')
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    
    # fitbit_path = os.path.join(save_dir, 'fitbit.parquet')
    # if not os.path.exists(fitbit_path):
    #     fitbit = get_fitbit_data()
    #     fitbit.to_parquet(fitbit_path)
    # else:
    #     fitbit = pd.read_parquet(fitbit_path)

    # questionare_path = os.path.join(save_dir, 'questionnaire.parquet')
    # prompts_path = os.path.join(save_dir, 'prompts.parquet')
    # if os.path.exists(questionare_path) and os.path.exists(prompts_path):
    #     feature_df = pd.read_parquet(questionare_path)
    #     prompts_df = pd.read_parquet(prompts_path)
    # else:
    #     feature_df, prompts_df = get_survey_data()
    #     feature_df.to_parquet(os.path.join(save_dir, 'questionnaire.parquet'))
    #     prompts_df.to_parquet(os.path.join(save_dir, 'prompts.parquet'))

    # fitbit = fitbit.rename(columns={'subject': 'participant_id'})
    # data = pd.merge(fitbit, feature_df, on=['participant_id', 'date'], how='inner')
    # data = pd.merge(data, prompts_df, on=['participant_id', 'date'], how='inner')
    # # add a column stress_label
    # data['stress_label'] = data['stress'].apply(binarize_stress)
    
    # invalid_indices = []
    # for i in range(len(data)):
    #     len_steps = data['steps'].iloc[i].shape[0]
    #     len_hr = data['heart_rate'].iloc[i].shape[0]
    #     len_cal = data['calories'].iloc[i].shape[0]
    #     len_dis = data['distance'].iloc[i].shape[0]
    #     # if any of them is not 24
    #     if len_steps != 24 or len_hr != 24 or len_cal != 24 or len_dis != 24:
    #         invalid_indices.append(i)
    # # remove rows in data at invalid_indices
    # data = data.drop(invalid_indices)
    
    # # join with participant data
    # participant_data = pd.DataFrame(PARTICIPANT_DATA).T.reset_index()
    # participant_data = participant_data.rename(columns={'index': 'participant_id'})
    # data = pd.merge(data, participant_data, on='participant_id', how='inner')
    # # ordinal encode the gender column
    # enc = OrdinalEncoder()
    # data['gender'] = enc.fit_transform(data['gender'].values.reshape(-1, 1))
    # # update prompts column
    # data['prompts'] = data.apply(update_prompts_column, axis=1)
    
    # data.to_parquet(os.path.join(save_dir, 'pmdata.parquet'))
    # # write column names as txt
    # with open(os.path.join(save_dir, 'pmdata.txt'), 'w') as f:
    #     f.write('\n'.join(data.columns))

    data = pd.read_parquet(os.path.join(save_dir, 'pmdata.parquet'))
    ids = data['participant_id'].unique()
    five_fold_ids = np.array_split(ids, 5)
    five_fold_train_test = {
        0: (np.concatenate(five_fold_ids[1:]), five_fold_ids[0]),
        1: (np.concatenate([five_fold_ids[0], five_fold_ids[2], five_fold_ids[3], five_fold_ids[4]]), five_fold_ids[1]),
        2: (np.concatenate([five_fold_ids[0], five_fold_ids[1], five_fold_ids[3], five_fold_ids[4]]), five_fold_ids[2]),
        3: (np.concatenate([five_fold_ids[0], five_fold_ids[1], five_fold_ids[2], five_fold_ids[4]]), five_fold_ids[3]),
        4: (np.concatenate(five_fold_ids[:-1]), five_fold_ids[4]),
    }
    
    def pmdata_train_test_split(df, n_splits=5, seed=42):
        # split
        for i, (train_ids, test_ids) in five_fold_train_test.items():
            print(f'split {i}/{n_splits}...')
            split_dir_path = os.path.join(save_dir, f'split_{i}')
            if not os.path.exists(split_dir_path):
                os.makedirs(split_dir_path)
            train = df[df['participant_id'].isin(train_ids)]
            test = df[df['participant_id'].isin(test_ids)]
            train.to_parquet(os.path.join(split_dir_path, 'train.parquet'))
            test.to_parquet(os.path.join(split_dir_path, 'test.parquet'))
            compute_stats_pmdata(os.path.join(split_dir_path, 'train.parquet'))
            compute_stats_pmdata(os.path.join(split_dir_path, 'test.parquet'))
        
    pmdata_train_test_split(data)

def exclude_attrs_from_desc(desc_values, exclude_attrs):
    updated_desc_values = []
    for desc in desc_values:
        # split desc by row
        desc = desc.split('\n')
        # exclude attributes
        desc = [d for d in desc if d.split(':')[0].strip() not in exclude_attrs]
        # join desc
        desc = '\n'.join(desc)
        updated_desc_values.append(desc)
    return np.array(updated_desc_values)

def txt_to_emb(desc_values, txt_encoder, batch_size=1024):
    # process in batches
    txt_emb = []
    for i in tqdm(range(0, len(desc_values), batch_size)):
        batch = desc_values[i:i + batch_size]
        txt_emb.append(txt_encoder(batch).cpu().numpy())
    txt_emb = np.concatenate(txt_emb, axis=0)
    return txt_emb

def save_text_emb():
    exclude_attrs_dict = {
        'demographics': ['age', 'gender', 'height'], 
        'food': ['meals', 'glasses of fluid', 'alcohol'], 
        'activity': ['activity'], 
        'sleep': ['sleep duration', 'sleep quality'], 
        'body': ['fatigue', 'readiness', 'soreness', 'injuries'], 
        'steps': ['steps'],
        'bpm': ['heart rate'],
        'calories': ['calories'],
        'distance': ['distance'],
    }
    txt_encoder = BertEncoder(device='cuda')
    saved_dir = os.path.join(PMDATA_PATH, 'processed')
    for fold in range(5):
        for split in ['train', 'test']:
            split_dir_path = os.path.join(saved_dir, f'split_{fold}')
            df = pd.read_parquet(os.path.join(split_dir_path, f'{split}.parquet'))
            
            # text embdding folder
            txt_emb_dir = os.path.join(split_dir_path, 'emb')
            if not os.path.exists(txt_emb_dir):
                os.makedirs(txt_emb_dir)
            
            desc_col = 'prompts'
            desc = df[desc_col].values
            txt_emb = txt_to_emb(desc, txt_encoder)
            np.save(os.path.join(txt_emb_dir, f'{split}_text_emb.npy'), txt_emb)
            for key, exclude_attrs in exclude_attrs_dict.items():
                updated_desc = exclude_attrs_from_desc(desc, exclude_attrs)
                txt_emb = txt_to_emb(updated_desc, txt_encoder)
                np.save(
                    os.path.join(txt_emb_dir, f'{split}_text_emb_no_{key}.npy'), 
                    txt_emb,
                )
            print(f'{split} in fold {fold} done')
            print()

def normalize_data():
    saved_dir = os.path.join(PMDATA_PATH, 'user_split')
    for fold in tqdm(range(5)):
        fold_dir = os.path.join(saved_dir, f'split_{fold}')
        train_df_pmdata = pd.read_parquet(
            os.path.join(
                fold_dir,
                'train.parquet',
            ),
        )
        test_df_pmdata = pd.read_parquet(
            os.path.join(
                fold_dir,
                'test.parquet',
            ),
        )
        for scaler in ['quantile']:
            for norm_type in ['global', 'user', 'date']:
                train_df_pmdata_norm, test_df_pmdata_norm = normalize_multimodal_x(
                    train_df_pmdata, 
                    test_df_pmdata, 
                    ['participant_id', 'date'], 
                    ['steps', 'heart_rate', 'calories', 'distance'], 
                    x_tab_num_cols=[
                        'BELOW_DEFAULT_ZONE_1', 'IN_DEFAULT_ZONE_1', 'IN_DEFAULT_ZONE_3',
                        'IN_DEFAULT_ZONE_2', 'lightly_active_minutes',
                        'moderately_active_minutes', 'sedentary_minutes', 'minutesAsleep',
                        'minutesAwake', 'timeInBed', 'efficiency', 'overall_score',
                        'composition_score', 'revitalization_score', 'deep_sleep_in_minutes',
                        'resting_heart_rate', 'restlessness', 'very_active_minutes',
                        'IN_CUSTOM_ZONE', 'BELOW_CUSTOM_ZONE', 'ABOVE_CUSTOM_ZONE', 'weight',
                        'glasses_of_fluid', 'perceived_exertion', 'duration_min', 'fatigue',
                        'mood', 'readiness', 'sleep_duration_h', 'sleep_quality', 'soreness',
                        'stress', 'age', 'height',
                    ],
                    scaler=scaler, 
                    norm_type=norm_type,
                )
                train_df_pmdata_norm.to_parquet(
                    os.path.join(
                        fold_dir,
                        f'train_{scaler}_{norm_type}.parquet',
                    ),
                    index=False,
                )
                test_df_pmdata_norm.to_parquet(
                    os.path.join(
                        fold_dir,
                        f'test_{scaler}_{norm_type}.parquet',
                    ),
                    index=False,
                )
                print(f'fold {fold} scaler {scaler} norm type {norm_type} done')

def regenerate_prompts_top_column(row):
    prompts = row['prompts']
    # delete the fitbit related rows in the original prompt
    prompts = prompts.split('\n')
    prompts = [d for d in prompts if d.split(':')[0].strip() not in ['steps', 'heart rate', 'calories', 'distance']]
    prompts = '\n'.join(prompts)
    
    description = []
    steps = row['steps']
    bpm = row['heart_rate']
    calories = row['calories']
    distance = row['distance']
    steps_stats = compute_hourly_stats(steps)  # return a dictionary keyed by stat type
    bpm_stats = compute_hourly_stats(bpm)
    calories_stats = compute_hourly_stats(calories)
    distance_stats = compute_hourly_stats(distance)
    steps_stats_str = f'steps: {convert_stats_to_str(steps_stats)}'
    bpm_stats_str = f'heart rate: {convert_stats_to_str(bpm_stats)}'
    calories_stats_str = f'calories: {convert_stats_to_str(calories_stats)}'
    distance_stats_str = f'distance: {convert_stats_to_str(distance_stats)}'
    description.append(steps_stats_str)
    description.append(bpm_stats_str)
    description.append(calories_stats_str)
    description.append(distance_stats_str)
    description = '\n'.join(description)
    description = description + '\n'
    
    prompts = description + prompts
    prompts = prompts.split('\n')
    # exclude attributes
    prompts = [d for d in prompts if d.split(':')[0].strip() not in ['stress', 'mood']]
    # join desc
    prompts = '\n'.join(prompts)

    return prompts

def run_regenerate():
    saved_dir = os.path.join(PMDATA_PATH, 'user_split')
    for fold in tqdm(range(5)):
        fold_dir = os.path.join(saved_dir, f'split_{fold}')
        for norm_type in ['global']:
            train_df_pmdata = pd.read_parquet(
                os.path.join(
                    fold_dir,
                    f'train_quantile_{norm_type}.parquet',
                ),
            )
            test_df_pmdata = pd.read_parquet(
                os.path.join(
                    fold_dir,
                    f'test_quantile_{norm_type}.parquet',
                ),
            )
            train_df_pmdata['prompts'] = train_df_pmdata.apply(regenerate_prompts_top_column, axis=1)
            test_df_pmdata['prompts'] = test_df_pmdata.apply(regenerate_prompts_top_column, axis=1)
            
            train_df_pmdata.to_parquet(
                os.path.join(
                    fold_dir,
                    f'train_quantile_{norm_type}.parquet',
                ),
                index=False,
            )
            test_df_pmdata.to_parquet(
                os.path.join(
                    fold_dir,
                    f'test_quantile_{norm_type}.parquet',
                ),
                index=False,
            )
            print(f'fold {fold} done')

def recompute_stats():
    saved_dir = os.path.join(PMDATA_PATH, 'user_split')
    for fold in range(5):
        fold_dir = os.path.join(saved_dir, f'split_{fold}')
        compute_stats_pmdata(os.path.join(fold_dir, 'train_quantile_global.parquet'))
        compute_stats_pmdata(os.path.join(fold_dir, 'test_quantile_global.parquet'))
        print(f'fold {fold} done')

def resave_text_emb():
    # relevant is everything except demographics, food, and activity
    exclude_attrs_dict = {
        'label': ['label'],
        # 'relevant': [
        #     'sleep duration', 'sleep quality',
        #     'fatigue', 'readiness', 'soreness', 'injuries',
        #     'steps', 'heart rate', 'calories', 'distance',
        #     'label',
        # ],
        # 'irrelevant': [
        #     'age', 'gender', 'height',
        #     'meals', 'glasses of fluid', 'alcohol',
        #     'activity',
        # ],
    }
    txt_encoder = BertEncoder(device='cuda')
    saved_dir = os.path.join(PMDATA_PATH, 'user_split')
    for fold in range(5):
        for split in ['train', 'test']:
            split_dir_path = os.path.join(saved_dir, f'split_{fold}')
            txt_emb_dir = os.path.join(split_dir_path, 'emb_quantile_global')
            if not os.path.exists(txt_emb_dir):
                os.makedirs(txt_emb_dir)
            
            desc_col = 'prompts'
            df = pd.read_parquet(
                os.path.join(split_dir_path, f'{split}_quantile_global.parquet'),
            )
            desc = df[desc_col].values
            
            txt_emb = txt_to_emb(desc, txt_encoder)
            np.save(os.path.join(txt_emb_dir, f'{split}_text_emb.npy'), txt_emb)

            for key, exclude_attrs in exclude_attrs_dict.items():
                updated_desc = exclude_attrs_from_desc(desc, exclude_attrs)
                txt_emb = txt_to_emb(updated_desc, txt_encoder)
                np.save(
                    os.path.join(txt_emb_dir, f'{split}_text_emb_no_{key}.npy'), 
                    txt_emb,
                )
            print(f'{split} in fold {fold} done')
            print()

if __name__ == '__main__':
    main()
    # save_text_emb()
    normalize_data()
    run_regenerate()
    recompute_stats()
    resave_text_emb()
    print('done')
