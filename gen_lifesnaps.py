import os
import ast
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from scipy import stats
from models import BertEncoder
import warnings
from utils import compute_hourly_stats, convert_stats_to_str
from utils import normalize_multimodal_x
from tqdm import tqdm
from constant import LIFESNAPS_PATH

warnings.filterwarnings('ignore')

def compute_stats_lifesnaps(df_path):
    stats_map = {
        'mean': np.mean,
        'std': np.std,
        'min': np.min,
        'max': np.max,
        '25': lambda x: np.percentile(x, 25),
        '50': np.median,
        '75': lambda x: np.percentile(x, 75),
        'rms': lambda x: np.sqrt(np.mean(x ** 2)),
        'kurtosis': stats.kurtosis,
        'skew': stats.skew,
        'iqr': stats.iqr,
    }
    df = pd.read_parquet(df_path)
    df_name = os.path.basename(df_path)
    df_name = df_name.split('.')[0]
    df_dir = os.path.dirname(df_path)
    
    # compute steps
    steps = df['steps']
    steps_stats = pd.DataFrame()
    for key, func in stats_map.items():
        steps_stats[key] = steps.apply(func)
    steps_stats.columns = [f'steps_{col}' for col in steps_stats.columns]

    # compute bpm
    bpm = df['bpm']
    bpm_stats = pd.DataFrame()
    for key, func in stats_map.items():
        bpm_stats[key] = bpm.apply(func)
    bpm_stats.columns = [f'bpm_{col}' for col in bpm_stats.columns]

    # compute temperature
    temperature = df['temperature']
    temperature_stats = pd.DataFrame()
    for key, func in stats_map.items():
        temperature_stats[key] = temperature.apply(func)
    temperature_stats.columns = [f'temperature_{col}' for col in temperature_stats.columns]

    # compute calories
    calories = df['calories']
    calories_stats = pd.DataFrame()
    for key, func in stats_map.items():
        calories_stats[key] = calories.apply(func)
    calories_stats.columns = [f'calories_{col}' for col in calories_stats.columns]

    # compute distance
    distance = df['distance']
    distance_stats = pd.DataFrame()
    for key, func in stats_map.items():
        distance_stats[key] = distance.apply(func)
    distance_stats.columns = [f'distance_{col}' for col in distance_stats.columns]

    # remove original columns
    df = df.drop(columns=['steps', 'bpm', 'temperature', 'calories', 'distance'])

    # first two columns, stats columns, and then the rest
    df = pd.concat(
        [
            df.iloc[:, :2], steps_stats, bpm_stats, temperature_stats, 
            calories_stats, distance_stats, df.iloc[:, 2:],
        ], 
        axis=1,
    )
    
    # save to parquet
    df.to_parquet(os.path.join(df_dir, f'{df_name}_stats.parquet'), index=False)

def lifesnaps_grouped_df_to_features(
    hourly_fitbit_sema_df_grouped, 
    fitbit_cols, 
    send_cols,
    personal_cols,
    activity_cols,
    location_cols,
):
    survey_cols = personal_cols + activity_cols + location_cols
    hourly_fitbit_features = []
    # remove duplicates in hour column for each group
    for name, group in hourly_fitbit_sema_df_grouped:
        if group['hour'].duplicated().any():
            # remove duplicates
            group = group.drop_duplicates(subset='hour')
        if group.shape[0] != 24:
            continue
        if not (group['hour'].values == list(range(24))).all():
            continue
        
        # fitbit
        fitbit = group[fitbit_cols]
        fitbit = fitbit.interpolate()
        fitbit = fitbit.fillna(0)
        
        # sendentary
        send = group[send_cols]
        send = send.sum()
        
        # tabular
        tabular = group[survey_cols]
        reduced_tabular = {}
        # pick the most common value except NaN for each column in tabular
        for column in tabular.columns:
            # if all NaN
            if tabular[column].isnull().all():
                reduced_tabular[column] = np.nan
            else:
                reduced_tabular[column] = tabular[column].mode()[0]

        # create a row
        row = {'id': name[0], 'date': name[1]}
        
        # time_series 24, 5, consider as 5 numpy arrays
        for column in fitbit.columns:
            row[column] = fitbit[column].values

        # extend sendentary
        for column in send_cols:
            row[column] = send[column]

        # extend survey
        for column in survey_cols:
            row[column] = reduced_tabular[column]

        # columns in location_cols where the value is 1
        location_vals = []
        for column in location_cols:
            if reduced_tabular[column] == 1.0:
                location_vals.append(column)
            else:
                pass
        row['location'] = location_vals if len(location_vals) > 0 else np.nan
        hourly_fitbit_features.append(row)

    hourly_fitbit_features = pd.DataFrame(hourly_fitbit_features)
    return hourly_fitbit_features

def get_desc_label(row, stress_label_type):
    assert stress_label_type in ['stress_original', 'stress_top', 'stress_sides', 'stress_40_60']

    description = []

    # fitbit
    steps = row['steps']
    bpm = row['bpm']
    calories = row['calories']
    distance = row['distance']
    temperature = row['temperature']

    # compute statstics for each fitbit modality and describe it
    steps_stats = compute_hourly_stats(steps)  # return a dictionary keyed by stat type
    bpm_stats = compute_hourly_stats(bpm)
    calories_stats = compute_hourly_stats(calories)
    distance_stats = compute_hourly_stats(distance)
    temperature_stats = compute_hourly_stats(temperature)

    steps_stats_str = f'steps: {convert_stats_to_str(steps_stats)}'
    bpm_stats_str = f'heart rate: {convert_stats_to_str(bpm_stats)}'
    calories_stats_str = f'calories: {convert_stats_to_str(calories_stats)}'
    distance_stats_str = f'distance: {convert_stats_to_str(distance_stats)}'
    temperature_stats_str = f'temperature: {convert_stats_to_str(temperature_stats)}'
    
    description.append(steps_stats_str)
    description.append(bpm_stats_str)
    description.append(calories_stats_str)
    description.append(distance_stats_str)
    description.append(temperature_stats_str)

    # tabular
    gender = row['gender']
    age = row['age']
    bmi = row['bmi']
    step_goal = row['step_goal_label']
    ipip_extraversion_category = row['ipip_extraversion_category']
    ipip_agreeableness_category = row['ipip_agreeableness_category']
    ipip_conscientiousness_category = row['ipip_conscientiousness_category']
    ipip_stability_category = row['ipip_stability_category']
    ipip_intellect_category = row['ipip_intellect_category']
    activity_type = row['activityType']
    location = row['location']
    stai_stress_category = row[stress_label_type]
    
    # gender
    if gender == 'MALE':
        gender_str = 'gender: male'
        description.append(gender_str)
    elif gender == 'FEMALE':
        gender_str = 'gender: female'
        description.append(gender_str)
    
    # age
    if age == '<30':
        age_str = 'age: younger than 30'
        description.append(age_str)
    elif age == '>=30':
        age_str = 'age: older than 30'
        description.append(age_str)
    
    # bmi
    if bmi == '<19':
        bmi_str = 'bmi: less than 19'
        description.append(bmi_str)
    elif bmi == '>=25':
        bmi_str = 'bmi: greater than 24'
        description.append(bmi_str)
    elif bmi == '>=30':
        bmi_str = 'bmi: greater than 29'
        description.append(bmi_str)
    
    # step goal
    if isinstance(step_goal, str):
        step_goal_str = f'step goal: {step_goal}'.lower()
        description.append(step_goal_str)

    # extraversion
    if isinstance(ipip_extraversion_category, str):
        extraversion_str = f'extraversion: {ipip_extraversion_category}'.lower()
        description.append(extraversion_str)
    
    # agreeableness
    if isinstance(ipip_agreeableness_category, str):
        agreeableness_str = f'agreeableness: {ipip_agreeableness_category}'.lower()
        description.append(agreeableness_str)

    # conscientiousness
    if isinstance(ipip_conscientiousness_category, str):
        conscientiousness_str = f'conscientiousness: {ipip_conscientiousness_category}'.lower()
        description.append(conscientiousness_str)

    # stability
    if isinstance(ipip_stability_category, str):
        stability_str = f'stability: {ipip_stability_category}'.lower()
        description.append(stability_str)

    # intellect
    if isinstance(ipip_intellect_category, str):
        intellect_str = f'intellect: {ipip_intellect_category}'.lower()
        description.append(intellect_str)

    # activity
    if isinstance(activity_type, str):
        activity = [act.lower() for act in ast.literal_eval(activity_type)]
        activity_str = f'activity: {", ".join(activity)}'
        description.append(activity_str)

    # location
    if isinstance(location, str):
        loc = [ele.lower() for ele in row['location']]
        loc = ', '.join(loc)
        loc_str = f'location: {loc}'
        description.append(loc_str)

    # stress
    if isinstance(stai_stress_category, str):
        stress_str = f'label: {stai_stress_category}'
        description.append(stress_str)

    description = '\n'.join(description)
    return description

def main():
    # save the processed data
    processed_dir = os.path.join(LIFESNAPS_PATH, 'processed')
    
    # # columns
    # fitbit_cols = ['temperature', 'calories', 'distance', 'bpm', 'steps']
    # send_cols = [
    #     'minutes_in_default_zone_1', 'minutes_below_default_zone_1', 
    #     'minutes_in_default_zone_2', 'minutes_in_default_zone_3',
    # ]
    # personal_cols = ['age', 'gender', 'bmi', 'step_goal_label']
    # activity_cols = ['activityType']
    location_cols = [
        'ENTERTAINMENT', 'GYM', 'HOME', 'HOME_OFFICE', 'OTHER', 
        'OUTDOORS', 'TRANSIT', 'WORK/SCHOOL',
    ]
    
    # read lifesnaps
    lifesnaps = pd.read_parquet(os.path.join(processed_dir, 'lifesnaps.parquet'))
    
    # save columns to txt
    with open(os.path.join(processed_dir, 'lifesnaps.txt'), 'w') as f:
        f.write('\n'.join(lifesnaps.columns.tolist()))
    
    # now, mutate the dataframe
    str_cols = [
        'age', 'gender', 'bmi', 'step_goal_label',
        'ipip_extraversion_category',
        'ipip_agreeableness_category',
        'ipip_conscientiousness_category',
        'ipip_stability_category',
        'ipip_intellect_category',
    ] + location_cols
    
    encoder = OrdinalEncoder()
    lifesnaps[str_cols] = encoder.fit_transform(lifesnaps[str_cols])

    # drop location and activityType because they are already in description
    # and they are not useful for the model
    lifesnaps.drop(
        columns=[
            'location', 'activityType',
            'stai_stress_category',
            'stress_original',
            'stress_top',
            'stress_sides',
            'stress_40_60',
        ], 
        inplace=True,
    )
    
    # write lifesnaps columns to txt
    with open(os.path.join(processed_dir, 'encoded.txt'), 'w') as f:
        f.write('\n'.join(lifesnaps.columns.tolist()))
    
    # fill NaN with -1
    lifesnaps.fillna(-1, inplace=True)
    
    # save to the encoded data to parquet and encoder to joblib
    lifesnaps.to_parquet(os.path.join(processed_dir, 'encoded.parquet'), index=False)
    encoder_path = os.path.join(processed_dir, 'encoded.joblib')
    joblib.dump(encoder, encoder_path)
    
    # original, top, sides, 40_60
    top_dir = os.path.join(processed_dir, 'user_split')
    for dir_path in [top_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    # print(lifesnaps.head())
    ids = lifesnaps['id'].unique()
    five_fold_ids = np.array_split(ids, 5)
    five_fold_train_test = {
        0: (np.concatenate(five_fold_ids[1:]), five_fold_ids[0]),
        1: (np.concatenate([five_fold_ids[0], five_fold_ids[2], five_fold_ids[3], five_fold_ids[4]]), five_fold_ids[1]),
        2: (np.concatenate([five_fold_ids[0], five_fold_ids[1], five_fold_ids[3], five_fold_ids[4]]), five_fold_ids[2]),
        3: (np.concatenate([five_fold_ids[0], five_fold_ids[1], five_fold_ids[2], five_fold_ids[4]]), five_fold_ids[3]),
        4: (np.concatenate(five_fold_ids[:-1]), five_fold_ids[4]),
    }
    # print(five_fold_train_test)

    def lifesnaps_train_test_split(df, n_splits=5, seed=42):
        label_dir_path = top_dir

        # split
        for i, (train_ids, test_ids) in five_fold_train_test.items():
            print(f'split {i}/{n_splits}...')
            split_dir_path = os.path.join(label_dir_path, f'split_{i}')
            if not os.path.exists(split_dir_path):
                os.makedirs(split_dir_path)
            train = df[df['id'].isin(train_ids)]
            test = df[df['id'].isin(test_ids)]
            train.to_parquet(os.path.join(split_dir_path, 'train.parquet'), index=False)
            test.to_parquet(os.path.join(split_dir_path, 'test.parquet'), index=False)
            compute_stats_lifesnaps(os.path.join(split_dir_path, 'train.parquet'))
            compute_stats_lifesnaps(os.path.join(split_dir_path, 'test.parquet'))

    lifesnaps_train_test_split(lifesnaps)

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
        'demographics': ['gender', 'age', 'bmi', 'step goal'], 
        'personality': ['extraversion', 'agreeableness', 'conscientiousness', 'stability', 'intellect'],
        'activity': ['activity'],
        'location': ['location'],
        'steps': ['steps'],
        'bpm': ['heart rate'],
        'calories': ['calories'],
        'distance': ['distance'],
        'temperature': ['temperature'],
    }

    txt_encoder = BertEncoder(device='cuda')
    label_dir_path = os.path.join(LIFESNAPS_PATH, 'processed', 'top')
    for fold in range(5):
        for split in ['train', 'test']:
            split_dir_path = os.path.join(label_dir_path, f'split_{fold}')
            df = pd.read_parquet(os.path.join(split_dir_path, f'{split}.parquet'))
            
            # text embdding folder
            txt_emb_dir = os.path.join(split_dir_path, 'emb')
            if not os.path.exists(txt_emb_dir):
                os.makedirs(txt_emb_dir)
            
            desc_col = 'desc_top'
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
            print(f'{split} done')
            print()

def normalize_data():
    saved_dir = os.path.join(LIFESNAPS_PATH, 'processed', 'user_split')
    for fold in tqdm(range(5)):
        fold_dir = os.path.join(saved_dir, f'split_{fold}')
        train_df_lifesnaps = pd.read_parquet(
            os.path.join(
                fold_dir,
                'train.parquet',
            ),
        )
        test_df_lifesnaps = pd.read_parquet(
            os.path.join(
                fold_dir,
                'test.parquet',
            ),
        )
        for scaler in ['quantile']:
            for norm_type in ['global', 'user', 'date']:
                train_df_lifesnaps_norm, test_df_lifesnaps_norm = normalize_multimodal_x(
                    train_df_lifesnaps, 
                    test_df_lifesnaps, 
                    ['id', 'date'], 
                    ['steps', 'bpm', 'calories', 'distance', 'temperature'], 
                    x_tab_num_cols=[
                        'minutes_in_default_zone_1', 'minutes_below_default_zone_1',
                        'minutes_in_default_zone_2', 'minutes_in_default_zone_3',
                    ],
                    scaler=scaler, 
                    norm_type=norm_type,
                )
                train_df_lifesnaps_norm.to_parquet(
                    os.path.join(
                        fold_dir,
                        f'train_{scaler}_{norm_type}.parquet',
                    ),
                    index=False,
                )
                test_df_lifesnaps_norm.to_parquet(
                    os.path.join(
                        fold_dir,
                        f'test_{scaler}_{norm_type}.parquet',
                    ),
                    index=False,
                )
                print(f'fold {fold} scaler {scaler} norm type {norm_type} done')

def regenerate_prompts_top_column(row):
    prompts = row['desc_top']
    # delete the fitbit related rows in the original prompt
    prompts = prompts.split('\n')
    prompts = [d for d in prompts if d.split(':')[0].strip() not in [
        'steps', 'heart rate', 'calories', 'distance', 'temperature',
    ]]
    prompts = '\n'.join(prompts)

    description = []
    steps = row['steps']
    bpm = row['bpm']
    calories = row['calories']
    distance = row['distance']
    temperature = row['temperature']

    steps_stats = compute_hourly_stats(steps)  # return a dictionary keyed by stat type
    bpm_stats = compute_hourly_stats(bpm)
    calories_stats = compute_hourly_stats(calories)
    distance_stats = compute_hourly_stats(distance)
    temperature_stats = compute_hourly_stats(temperature)

    steps_stats_str = f'steps: {convert_stats_to_str(steps_stats)}'
    bpm_stats_str = f'heart rate: {convert_stats_to_str(bpm_stats)}'
    calories_stats_str = f'calories: {convert_stats_to_str(calories_stats)}'
    distance_stats_str = f'distance: {convert_stats_to_str(distance_stats)}'
    temperature_stats_str = f'temperature: {convert_stats_to_str(temperature_stats)}'

    description.append(steps_stats_str)
    description.append(bpm_stats_str)
    description.append(calories_stats_str)
    description.append(distance_stats_str)
    description.append(temperature_stats_str)

    description = '\n'.join(description)
    description = description + '\n'
    
    prompts = description + prompts

    return prompts

def run_regenerate():
    saved_dir = os.path.join(LIFESNAPS_PATH, 'processed', 'user_split')
    for fold in tqdm(range(5)):
        fold_dir = os.path.join(saved_dir, f'split_{fold}')
        for norm_type in ['global']:
            train_df_lifesnaps = pd.read_parquet(
                os.path.join(
                    fold_dir,
                    f'train_quantile_{norm_type}.parquet',
                ),
            )
            test_df_lifesnaps = pd.read_parquet(
                os.path.join(
                    fold_dir,
                    f'test_quantile_{norm_type}.parquet',
                ),
            )
            train_df_lifesnaps['desc_top'] = train_df_lifesnaps.apply(
                regenerate_prompts_top_column, axis=1,
            )
            test_df_lifesnaps['desc_top'] = test_df_lifesnaps.apply(
                regenerate_prompts_top_column, axis=1,
            )
            train_df_lifesnaps.to_parquet(
                os.path.join(
                    fold_dir,
                    f'train_quantile_{norm_type}.parquet',
                ),
                index=False,
            )
            test_df_lifesnaps.to_parquet(
                os.path.join(
                    fold_dir,
                    f'test_quantile_{norm_type}.parquet',
                ),
                index=False,
            )
            print(f'fold {fold} norm type {norm_type} done')

def recompute_stats():
    saved_dir = os.path.join(LIFESNAPS_PATH, 'processed', 'user_split')
    for fold in tqdm(range(5)):
        fold_dir = os.path.join(saved_dir, f'split_{fold}')
        for norm_type in ['global']:
            compute_stats_lifesnaps(os.path.join(fold_dir, f'train_quantile_{norm_type}.parquet'))
            compute_stats_lifesnaps(os.path.join(fold_dir, f'test_quantile_{norm_type}.parquet'))
            print(f'fold {fold} norm type {norm_type} done')

def resave_text_emb():    
    # relevant is everything except demographics
    exclude_attrs_dict = {
        'label': ['label'],
        # 'relevant': [
        #     'extraversion', 'agreeableness', 'conscientiousness', 'stability', 'intellect',
        #     'activity'
        #     'location',
        #     'steps', 'heart rate', 'calories', 'distance', 'temperature',
        #     'label',
        # ],
        # 'irrelevant': [
        #     'gender', 'age', 'bmi', 'step goal',
        # ],
    }
    txt_encoder = BertEncoder(device='cuda')
    label_dir_path = os.path.join(LIFESNAPS_PATH, 'processed', 'user_split')
    for fold in range(5):
        for split in ['train', 'test']:
            split_dir_path = os.path.join(label_dir_path, f'split_{fold}')
            df = pd.read_parquet(os.path.join(split_dir_path, f'{split}_quantile_global.parquet'))

            # text embdding folder
            txt_emb_dir = os.path.join(split_dir_path, 'emb_quantile_global')
            if not os.path.exists(txt_emb_dir):
                os.makedirs(txt_emb_dir)

            desc_col = 'desc_top'
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
            print(f'{split} done')
            print()

if __name__ == '__main__':
    # main()
    # save_text_emb()
    # normalize_data()
    # run_regenerate()
    # recompute_stats()
    resave_text_emb()
    print('done')
