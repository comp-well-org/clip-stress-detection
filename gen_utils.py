import os
import numpy as np
import pandas as pd
from scipy import stats
from torch.utils.data import DataLoader, Dataset
from constant import LIFESNAPS_PATH, PMDATA_PATH
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import QuantileTransformer, FunctionTransformer
import warnings

warnings.filterwarnings('ignore')

def get_normalizer(scaler, n_examples=0, seed=2024):
    if scaler == 'standard':
        normalizer = StandardScaler()
    elif scaler == 'minmax':
        normalizer = MinMaxScaler()
    elif scaler == 'quantile':
        slices = 30
        normalizer = QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(n_examples // slices, 1000), 10),
            subsample=10 ** 9,
            random_state=seed,
        )
    elif scaler == 'none':
        normalizer = FunctionTransformer()
    return normalizer

def normalize_multimodal_x(
    train_df, test_df, id_date_cols, fitbit_cols, 
    x_tab_num_cols=None, scaler='minmax', norm_type='global',
):
    # make copies of the dataframes to avoid modifying the original data
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    id_col, date_col = id_date_cols
    ids_train = train_df[id_col].unique()
    dates_train = train_df[date_col].unique()
    
    ids_test = test_df[id_col].unique()
    dates_test = test_df[date_col].unique()
    
    if norm_type == 'global':
        for col in fitbit_cols:
            n_examples = len(train_df) * 3
            normalizer = get_normalizer(scaler, n_examples)
            train_col_values = np.stack(train_df[col])
            test_col_values = np.stack(test_df[col])
            normalizer.fit(train_col_values)
            train_col_values_norm = normalizer.transform(train_col_values)
            test_col_values_norm = normalizer.transform(test_col_values)
            train_df[col] = [row for row in train_col_values_norm]
            test_df[col] = [row for row in test_col_values_norm]
        for col in x_tab_num_cols:
            n_examples = len(train_df) 
            normalizer = get_normalizer(scaler, n_examples)
            train_col_values = train_df[col].values.reshape(-1, 1)
            test_col_values = test_df[col].values.reshape(-1, 1)
            normalizer.fit(train_col_values)
            train_col_values_norm = normalizer.transform(train_col_values)
            test_col_values_norm = normalizer.transform(test_col_values)
            train_df[col] = train_col_values_norm
            test_df[col] = test_col_values_norm
    elif norm_type == 'user' or norm_type == 'date':
        if norm_type == 'user':
            iter_list_train = ids_train
            iter_list_test = ids_test
            iter_col = id_col
        elif norm_type == 'date':
            iter_list_train = dates_train
            iter_list_test = dates_test
            iter_col = date_col
        
        for col in fitbit_cols:
            dict_of_normalizers = {}
            for iter_item in iter_list_train:
                user_data = np.stack(train_df[train_df[iter_col] == iter_item][col])
                n_examples = len(user_data) * 3
                normalizer = get_normalizer(scaler, n_examples)
                normalizer.fit(user_data)
                dict_of_normalizers[iter_item] = normalizer
                user_data_norm = normalizer.transform(user_data)
                uid_indices = train_df[iter_col] == iter_item
                uid_indices = np.where(uid_indices)[0]
                for i in range(len(uid_indices)):
                    train_df.iloc[uid_indices[i]][col] = user_data_norm[i]
            for iter_item in iter_list_test:
                user_data = np.stack(test_df[test_df[iter_col] == iter_item][col])
                if iter_item in dict_of_normalizers:
                    normalizer = dict_of_normalizers[iter_item]
                    user_data_norm = normalizer.transform(user_data)
                    uid_indices = test_df[iter_col] == iter_item
                    uid_indices = np.where(uid_indices)[0]
                    for i in range(len(uid_indices)):
                        test_df.iloc[uid_indices[i]][col] = user_data_norm[i]
                else:
                    # ensemble of normalizers
                    user_data_norm_list = []
                    for normalizer in dict_of_normalizers.values():
                        user_data_normalized = normalizer.transform(user_data)
                        user_data_norm_list.append(user_data_normalized)
                    # average the normalized data
                    user_data_norm = np.mean(user_data_norm_list, axis=0)
                    uid_indices = test_df[iter_col] == iter_item
                    uid_indices = np.where(uid_indices)[0]
                    for i in range(len(uid_indices)):
                        test_df.iloc[uid_indices[i]][col] = user_data_norm[i]
        for col in x_tab_num_cols:
            dict_of_normalizers = {}
            for iter_item in iter_list_train:
                user_data = train_df[train_df[iter_col] == iter_item][col].values.reshape(-1, 1)
                n_examples = len(user_data)
                normalizer = get_normalizer(scaler, n_examples)
                normalizer.fit(user_data)
                dict_of_normalizers[iter_item] = normalizer
                user_data_norm = normalizer.transform(user_data)
                uid_indices = train_df[iter_col] == iter_item
                uid_indices = np.where(uid_indices)[0]
                for i in range(len(uid_indices)):
                    train_df.iloc[uid_indices[i]][col] = user_data_norm[i]
            for iter_item in iter_list_test:
                user_data = test_df[test_df[iter_col] == iter_item][col].values.reshape(-1, 1)
                if iter_item in dict_of_normalizers:
                    normalizer = dict_of_normalizers[iter_item]
                    user_data_norm = normalizer.transform(user_data)
                    uid_indices = test_df[iter_col] == iter_item
                    uid_indices = np.where(uid_indices)[0]
                    for i in range(len(uid_indices)):
                        test_df.iloc[uid_indices[i]][col] = user_data_norm[i]
                else:
                    # ensemble of normalizers
                    user_data_norm_list = []
                    for normalizer in dict_of_normalizers.values():
                        user_data_normalized = normalizer.transform(user_data)
                        user_data_norm_list.append(user_data_normalized)
                    # average the normalized data
                    user_data_norm = np.mean(user_data_norm_list, axis=0)
                    uid_indices = test_df[iter_col] == iter_item
                    uid_indices = np.where(uid_indices)[0]
                    for i in range(len(uid_indices)):
                        test_df.iloc[uid_indices[i]][col] = user_data_norm[i]
    
    return train_df, test_df

def compute_hourly_stats(hourly_x):
    """Compute statistics for hourly time series."""
    stats_map = {
        'mean': np.mean,
        'standard deviation': np.std,
        'min': np.min,
        'max': np.max,
        'median': np.median,
        'root mean squared': lambda x: np.sqrt(np.mean(x ** 2)),
        'kurtosis': stats.kurtosis,
        'skewness': stats.skew,
        'interquartile range': stats.iqr,
    }
    data_stats = {}
    for stat_name, stat_func in stats_map.items():
        # round to one decimal place
        data_stats[stat_name] = round(stat_func(hourly_x), 1)
    return data_stats

def convert_stats_to_str(stats_dict):
    """Convert dictionary of statistics to a string."""
    stats_str_list = []
    for stat_name, stat_val in stats_dict.items():
        stats_str = f'{stat_name} is {stat_val}'
        stats_str_list.append(stats_str)
    stats_str = ', '.join(stats_str_list)
    return stats_str

def list_files(dir_path: str, ext: str = None) -> list:
    if ext is not None:
        abs_path_lst = []
        for files in os.listdir(dir_path):
            if files.endswith(ext):
                path = os.path.join(dir_path, files)
                if os.path.isfile(path):
                    abs_path_lst.append(path)
        return sorted(abs_path_lst)
    abs_path_lst = []
    for files in os.listdir(dir_path):
        path = os.path.join(dir_path, files)
        if os.path.isfile(path):
            abs_path_lst.append(path)
    return sorted(abs_path_lst)

class AvgMeter:
    def __init__(self, name='Metric'):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f'{self.name}: {self.avg:.4f}'
        return text

class LifeSnapsDataset(Dataset):
    # NOTE: first channel in fitbit is steps, second channel is heart rate
    def __init__(
        self, lifesnaps_df, cols, text_emb, 
        unlabeled_proportion=0.2, 
        labeled_proportion=1.0,
        seed=42,
        id=False,
    ):
        self.id = id
        self.df = lifesnaps_df
        self.cols = cols
        self.key_cols = cols[:2]
        self.fitbit_cols = ['steps', 'bpm', 'calories', 'distance', 'temperature']
        tab_end_idx = cols.index('ipip_intellect_category') + 1
        self.tabular_cols = cols[7:tab_end_idx]
        stress_label_col = 'stress_top_y'
        
        # write fitbit cols as txt
        with open(os.path.join(LIFESNAPS_PATH, 'processed', 'fitbit_cols.txt'), 'w') as f:
            for item in self.fitbit_cols:
                f.write(f'{item}\n')

        # write tabular cols as txt
        with open(os.path.join(LIFESNAPS_PATH, 'processed', 'tabular_cols.txt'), 'w') as f:
            for item in self.tabular_cols:
                f.write(f'{item}\n')
        
        # random seed
        np.random.seed(seed)
        
        # labeled indices
        labeled_indices = lifesnaps_df.index[lifesnaps_df[stress_label_col] != -1].tolist()
        unlabeled_indices = lifesnaps_df.index[lifesnaps_df[stress_label_col] == -1].tolist()
        
        chosen_labeled_indices = np.random.choice(
            labeled_indices, int(len(labeled_indices) * (1 - labeled_proportion)),
        )
        lifesnaps_df[stress_label_col].iloc[chosen_labeled_indices] = -1
        
        # sample labeled indices
        unlabeled_indices = np.random.choice(
            unlabeled_indices, int(len(unlabeled_indices) * unlabeled_proportion),
        )
        
        sampled_indices = np.concatenate([labeled_indices, unlabeled_indices])
        self.df = lifesnaps_df.iloc[sampled_indices]
        
        self.fitbit = self.df[self.fitbit_cols]
        self.fitbit = self.fitbit.apply(lambda x: np.stack(x), axis=1)
        self.fitbit = np.stack(self.fitbit)
        # fill na in the fitbit data with 0
        self.fitbit = np.nan_to_num(self.fitbit)
        
        if self.id:
            self.tabular = self.df[self.tabular_cols + ['id']].fillna(0).values
        else:
            self.tabular = self.df[self.tabular_cols].fillna(0).values
        self.stress = self.df[stress_label_col].values

        self.desc = text_emb[sampled_indices]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        fitbit = self.fitbit[idx]
        tabular = self.tabular[idx]
        stress = self.stress[idx]
        desc = self.desc[idx]
        return fitbit, tabular, stress, desc

class PMDataDataset(Dataset):
    def __init__(
        self, pmdata_df, text_emb, 
        unlabeled_proportion=0.2, labeled_proportion=1.0, 
        seed=42,
        id=False,
    ):
        # print('pmdata_df shape:', pmdata_df.shape)
        # print('text_emb shape:', text_emb.shape)
        self.id = id
        self.df = pmdata_df
        self.cols = self.df.columns.tolist()
        self.key_cols = self.cols[:2]  # participant_id, date
        self.fitbit_cols = ['steps', 'heart_rate', 'calories', 'distance']
        self.tabular_cols = [
            # location
            'BELOW_DEFAULT_ZONE_1', 'IN_DEFAULT_ZONE_1', 'IN_DEFAULT_ZONE_3', 'IN_DEFAULT_ZONE_2', 
            # active
            'lightly_active_minutes', 'moderately_active_minutes', 
            'sedentary_minutes', 
            'very_active_minutes', 
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
            # sleep
            'minutesAsleep', 'minutesAwake', 'timeInBed',
            'deep_sleep_in_minutes', 'sleep_duration_h', 'sleep_quality', 
            # body
            'soreness', 'fatigue', 'readiness', 'hadInjury', 'restlessness',
            # eat
            'hadBreakfast', 'hadLunch', 'hadDinner', 'hadEvening',
        ]
        self.stress_label_col = 'stress_label'
        np.random.seed(seed)
        
        # write fitbit cols as txt
        with open(os.path.join(PMDATA_PATH, 'processed', 'fitbit_cols.txt'), 'w') as f:
            for item in self.fitbit_cols:
                f.write(f'{item}\n')
        
        # write tabular cols as txt
        with open(os.path.join(PMDATA_PATH, 'processed', 'tabular_cols.txt'), 'w') as f:
            for item in self.tabular_cols:
                f.write(f'{item}\n')
        
        # labeled indices
        labeled_indices = pmdata_df.index[pmdata_df[self.stress_label_col] != -1].tolist()
        unlabeled_indices = pmdata_df.index[pmdata_df[self.stress_label_col] == -1].tolist()
        
        chosen_labeled_indices = np.random.choice(
            labeled_indices, int(len(labeled_indices) * (1 - labeled_proportion)),
        )
        pmdata_df[self.stress_label_col].iloc[chosen_labeled_indices] = -1
        
        # sample labeled indices
        unlabeled_indices = np.random.choice(
            unlabeled_indices, int(len(unlabeled_indices) * unlabeled_proportion),
        )

        sampled_indices = np.concatenate([labeled_indices, unlabeled_indices])
        self.df = pmdata_df.iloc[sampled_indices]
        
        self.fitbit = self.df[self.fitbit_cols]        
        self.fitbit = self.fitbit.apply(
            lambda x: np.stack(x), axis=1,
        )
        self.fitbit = np.stack(self.fitbit)
        # fill na in the fitbit data with 0
        self.fitbit = np.nan_to_num(self.fitbit)
        
        if self.id:
            self.tabular = self.df[self.tabular_cols + ['participant_id']]
        else:
            self.tabular = self.df[self.tabular_cols]
        object_cols = [
            'lightly_active_minutes', 'moderately_active_minutes', 
            'sedentary_minutes', 'very_active_minutes',
        ]
        # these columns are object type, convert to float
        self.tabular[object_cols] = self.tabular[object_cols].astype(float)
        self.tabular = self.tabular.fillna(0).values
        
        self.stress = self.df[self.stress_label_col].values
        
        # print('tabular shape:', self.tabular.shape)
        # print('text_emb shape:', text_emb.shape)
        
        self.desc = text_emb[sampled_indices]
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        fitbit = self.fitbit[idx]
        tabular = self.tabular[idx]
        stress = self.stress[idx]
        desc = self.desc[idx]
        return fitbit, tabular, stress, desc

def get_pmdata_loader(
    flag, batch_size=256, exclude: str = 'none', fold: int = 0,
    unlabeled_proportion=0.2, labeled_proportion=1.0,
    seed=42, scaler='none', norm_type='none',
    id=False,
    user_split=False,
):
    assert scaler in ['standard', 'minmax', 'quantile', 'none']
    assert norm_type in ['global', 'user', 'date', 'none']
    
    if flag == 'test':
        unlabeled_proportion = 0.0
        labeled_proportion = 1.0
    
    exclude_attrs_lists = [
        'demographics', 'food', 'activity', 'sleep', 'body',
        'steps', 'bpm', 'calories', 'distance', 'label',
        'relevant', 'irrelevant', 'all',
    ]
    if exclude != 'none':
        assert exclude in exclude_attrs_lists
    if exclude != 'none' and exclude != 'all':
        exd = '_'.join(exclude.split())
        emb_txt = f'text_emb_no_{exd}.npy'        
    else:
        emb_txt = 'text_emb.npy'
    
    if id:
        id_str = '_id'
    else:
        id_str = ''

    if user_split:
        folder_name = 'user_split'
    else:
        folder_name = 'processed'
    
    if scaler == 'none' and norm_type == 'none':
        train_df_filename = f'train{id_str}.parquet'
        test_df_filename = f'test{id_str}.parquet'
        emb_dirname = 'emb'
    else:
        train_df_filename = f'train_{scaler}_{norm_type}{id_str}.parquet'
        test_df_filename = f'test_{scaler}_{norm_type}{id_str}.parquet'
        emb_dirname = f'emb_{scaler}_{norm_type}'
        
    if flag == 'train':
        file_path = os.path.join(PMDATA_PATH, folder_name, f'split_{fold}', train_df_filename)
        emb_path = os.path.join(
            PMDATA_PATH, folder_name, f'split_{fold}/{emb_dirname}', f'train_{emb_txt}',
        )
        shuffle = True
    elif flag == 'test':
        file_path = os.path.join(PMDATA_PATH, folder_name, f'split_{fold}', test_df_filename)
        emb_path = os.path.join(
            PMDATA_PATH, folder_name, f'split_{fold}/{emb_dirname}', f'test_{emb_txt}',
        )
        shuffle = False
        
    pmdata = pd.read_parquet(file_path)
    text_emb = np.load(emb_path)
    if exclude == 'all':
        text_emb = np.random.randn(*text_emb.shape)
    pmdata_dataset = PMDataDataset(
        pmdata, text_emb, 
        unlabeled_proportion=unlabeled_proportion, 
        labeled_proportion=labeled_proportion, 
        seed=seed,
        id=id,
    )
    pmdata_loader = DataLoader(pmdata_dataset, batch_size=batch_size, shuffle=shuffle)
    return pmdata_loader

def get_lifesnaps_loader(
    flag, batch_size=256, exclude: str = 'none', fold: int = 0,
    unlabeled_proportion=0.2, 
    labeled_proportion=1.0,
    seed=42, scaler='none', norm_type='none',
    id=False,
    user_split=False,
):
    assert scaler in ['standard', 'minmax', 'quantile', 'none']
    assert norm_type in ['global', 'user', 'date', 'none']
    
    if flag == 'test':
        unlabeled_proportion = 0.0
        labeled_proportion = 1.0
    
    exclude_attrs_lists = [
        'demographics', 'personality', 'activity', 'location',
        'steps', 'bpm', 'calories', 'distance', 'temperature',
        'label', 'relevant', 'irrelevant', 'all',
    ]
    if exclude != 'none':
        assert exclude in exclude_attrs_lists
    if exclude != 'none' and exclude != 'all':
        exd = '_'.join(exclude.split())
        emb_txt = f'text_emb_no_{exd}.npy'
    else:
        emb_txt = 'text_emb.npy'
        
    if id:
        id_str = '_id'
    else:
        id_str = ''
        
    if scaler == 'none' and norm_type == 'none':
        train_df_filename = f'train{id_str}.parquet'
        test_df_filename = f'test{id_str}.parquet'
        emb_dirname = 'emb'
    else:
        train_df_filename = f'train_{scaler}_{norm_type}{id_str}.parquet'
        test_df_filename = f'test_{scaler}_{norm_type}{id_str}.parquet'
        emb_dirname = f'emb_{scaler}_{norm_type}'

    if user_split:
        folder_name = 'user_split'
    else:
        folder_name = 'top'

    if flag == 'train':
        file_path = os.path.join(
            LIFESNAPS_PATH, f'processed/{folder_name}', f'split_{fold}', train_df_filename,
        )
        emb_path = os.path.join(
            LIFESNAPS_PATH, f'processed/{folder_name}', f'split_{fold}/{emb_dirname}', f'train_{emb_txt}',
        )
        shuffle = True
    elif flag == 'test':
        file_path = os.path.join(
            LIFESNAPS_PATH, f'processed/{folder_name}', f'split_{fold}', test_df_filename,
        )
        emb_path = os.path.join(
            LIFESNAPS_PATH, f'processed/{folder_name}', f'split_{fold}/{emb_dirname}', f'test_{emb_txt}',
        )
        shuffle = False
    
    lifesnaps = pd.read_parquet(file_path)
    with open(os.path.join(LIFESNAPS_PATH, 'processed', 'encoded.txt')) as f:
        lifesnaps_cols = f.read().splitlines()
    text_emb = np.load(emb_path)
    if exclude == 'all':
        text_emb = np.random.randn(*text_emb.shape)
    lifesnaps_dataset = LifeSnapsDataset(
        lifesnaps, lifesnaps_cols, text_emb,
        unlabeled_proportion, 
        labeled_proportion,
        seed,
        id=id,
    )
    lifesnaps_loader = DataLoader(lifesnaps_dataset, batch_size=batch_size, shuffle=shuffle)
    return lifesnaps_loader
