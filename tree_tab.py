import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from utils import get_lifesnaps_stats, get_pmdata_stats

tab_cols_pmdata = [
    'BELOW_DEFAULT_ZONE_1',
    'IN_DEFAULT_ZONE_1',
    'IN_DEFAULT_ZONE_3',
    'IN_DEFAULT_ZONE_2',
    'lightly_active_minutes',
    'moderately_active_minutes',
    'sedentary_minutes',
    'very_active_minutes',
    'efficiency',
    'resting_heart_rate',
    'overall_score',
    'composition_score',
    'revitalization_score',
    'IN_CUSTOM_ZONE',
    'BELOW_CUSTOM_ZONE',
    'ABOVE_CUSTOM_ZONE',
    'age',
    'weight',
    'height',
    'gender',
    'glasses_of_fluid',
    'perceived_exertion',
    'duration_min',
    'minutesAsleep',
    'minutesAwake',
    'timeInBed',
    'deep_sleep_in_minutes',
    'sleep_duration_h',
    'sleep_quality',
    'soreness',
    'fatigue',
    'readiness',
    'hadInjury',
    'restlessness',
    'hadBreakfast',
    'hadLunch',
    'hadDinner',
    'hadEvening',
]

tab_cols_lifesnaps = [
    'minutes_in_default_zone_1',    
    'minutes_below_default_zone_1',
    'minutes_in_default_zone_2',
    'minutes_in_default_zone_3',
    'age',
    'gender',
    'bmi',
    'step_goal_label',
    'ENTERTAINMENT',
    'GYM',
    'HOME',
    'HOME_OFFICE',
    'OTHER',
    'OUTDOORS',
    'TRANSIT',
    'WORK/SCHOOL',
    'ipip_extraversion_category',
    'ipip_agreeableness_category',
    'ipip_conscientiousness_category',
    'ipip_stability_category',
    'ipip_intellect_category',
]

def bfs(ds_name, clf_name='xgboost', fold=0, seed=0):
    if ds_name == 'lifesnaps':
        x_train, y_train = get_lifesnaps_stats(
            flag='train', norm_type='quantile_global', fold=fold,
        )
        x_eval, y_eval = get_lifesnaps_stats(
            flag='test', norm_type='quantile_global', fold=fold,
        )
    elif ds_name == 'pmdata':
        x_train, y_train = get_pmdata_stats(
            flag='train', norm_type='quantile_global', fold=fold,
        )
        x_eval, y_eval = get_pmdata_stats(
            flag='test', norm_type='quantile_global', fold=fold,
        )

    # only consider indices where y is not -1
    x_train = x_train[y_train != -1]
    y_train = y_train[y_train != -1]
    x_eval = x_eval[y_eval != -1]
    y_eval = y_eval[y_eval != -1]
    
    if ds_name == 'lifesnaps':
        x_train = x_train[tab_cols_lifesnaps]
        x_eval = x_eval[tab_cols_lifesnaps]
    elif ds_name == 'pmdata':
        x_train = x_train[tab_cols_pmdata]
        x_eval = x_eval[tab_cols_pmdata]

    clf_map = {
        'random forest': RandomForestClassifier(random_state=seed),
        'xgboost': XGBClassifier(seed=seed),
    }

    clf = clf_map[clf_name]
    clf.fit(x_train, y_train)
    yhat_eval = clf.predict(x_eval)
    acc = accuracy_score(y_eval, yhat_eval)
    auc = roc_auc_score(y_eval, yhat_eval)
    return acc, auc

def main():
    results = []
    dataset_list = ['lifesnaps', 'pmdata']
    for ds_name in dataset_list:
        for clf_name in ['xgboost', 'random forest']:
            for fold in range(5):
                for seed in range(4):
                    acc, auc = bfs(ds_name, clf_name, fold, seed)
                    results.append({
                        'dataset': ds_name,
                        'clf': clf_name,
                        'fold': fold,
                        'seed': seed,
                        'acc': acc,
                        'auc': auc,
                    })
    results_df = pd.DataFrame(results)
    # save results as csv
    results_df.to_csv('results_tab.csv', index=False)

if __name__ == '__main__':
    main()
