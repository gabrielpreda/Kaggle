# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy.stats import skew
import time
import gc
gc.enable()


IS_LOCAL = True

import os

if(IS_LOCAL):
    PATH="../input/santander-value-prediction-challenge/"
else:
    PATH="../input/"

print(os.listdir(PATH))    
print(lgb.__version__)

def get_selected_features():
    return [
        'f190486d6', 'c47340d97', 'eeb9cd3aa', '66ace2992', 'e176a204a',
        '491b9ee45', '1db387535', 'c5a231d81', '0572565c2', '024c577b9',
        '15ace8c9f', '23310aa6f', '9fd594eec', '58e2e02e6', '91f701ba2',
        'adb64ff71', '2ec5b290f', '703885424', '26fc93eb7', '6619d81fc',
        '0ff32eb98', '70feb1494', '58e056e12', '1931ccfdd', '1702b5bf0',
        '58232a6fb', '963a49cdc', 'fc99f9426', '241f0f867', '5c6487af1',
        '62e59a501', 'f74e8f13d', 'fb49e4212', '190db8488', '324921c7b',
        'b43a7cfd5', '9306da53f', 'd6bb78916', 'fb0f5dbfe', '6eef030c1'
    ]

def get_data():
    print('Reading data')
    data = pd.read_csv(PATH+'train.csv', nrows=None)
    test = pd.read_csv(PATH+'test.csv', nrows=None)
    print('Train shape ', data.shape, ' Test shape ', test.shape)
    return data, test

def add_statistics(data, test):
    # This is part of the trick I think, plus lightgbm has a special process for NaNs
    data.replace(0, np.nan, inplace=True)
    test.replace(0, np.nan, inplace=True)
    
    original_features = [f for f in data.columns if f not in ['target', 'ID']]
    for df in [data, test]:
        df['nb_nans'] = df[original_features].isnull().sum(axis=1)
        # All of the stats will be computed without the 0s 
        df['the_median'] = df[original_features].median(axis=1)
        df['the_mean'] = df[original_features].mean(axis=1)
        df['the_sum'] = df[original_features].sum(axis=1)
        df['the_std'] = df[original_features].std(axis=1)
        df['the_kur'] = df[original_features].kurtosis(axis=1)
        
    return data, test


def fit_predict(data, y, test):
    # Get the features we're going to train on 
    features = get_selected_features() + ['nb_nans', 'the_median', 'the_mean', 'the_sum', 'the_std', 'the_kur']
    # Create folds
    folds = KFold(n_splits=5, shuffle=True, random_state=1)
    # Convert to lightgbm Dataset
    dtrain = lgb.Dataset(data=data[features], label=np.log1p(y['target']), free_raw_data=False)
    # Construct dataset so that we can use slice()
    dtrain.construct()
    # Init predictions
    sub_preds = np.zeros(test.shape[0])
    oof_preds = np.zeros(data.shape[0])
    # Lightgbm parameters
    # Optimized version scores 0.40
    # Step |   Time |      Score |      Stdev |   p1_leaf |   p2_subsamp |   p3_colsamp |   p4_gain |   p5_alph |   p6_lamb |   p7_weight |
    #   41 | 00m04s |   -1.36098 |    0.02917 |    9.2508 |       0.7554 |       0.7995 |   -3.3108 |   -0.1635 |   -0.9460 |      0.6485 |
    lgb_params = {
        'objective': 'regression',
        'num_leaves': 58,
        'subsample': 0.6143,
        'colsample_bytree': 0.6453,
        'min_split_gain': np.power(10, -2.5988),
        'reg_alpha': np.power(10, -2.2887),
        'reg_lambda': np.power(10, 1.7570),
        'min_child_weight': np.power(10, -0.1477),
        'verbose': -1,
        'seed': 3,
        'boosting_type': 'gbdt',
        'max_depth': 10,
        'learning_rate': 0.05,
        'metric': 'l2',
    }
    # Run KFold
    for trn_idx, val_idx in folds.split(data):
        # Train lightgbm
        clf = lgb.train(
            params=lgb_params,
            train_set=dtrain.subset(trn_idx),
            valid_sets=dtrain.subset(val_idx),
            num_boost_round=10000, 
            early_stopping_rounds=100,
            verbose_eval=50
        )
        # Predict Out Of Fold and Test targets
        # Using lgb.train, predict will automatically select the best round for prediction
        oof_preds[val_idx] = clf.predict(dtrain.data.iloc[val_idx])
        sub_preds += clf.predict(test[features]) / folds.n_splits
        # Display current fold score
        print(mean_squared_error(np.log1p(y['target'].iloc[val_idx]), 
                                 oof_preds[val_idx]) ** .5)
    # Display Full OOF score (square root of a sum is not the sum of square roots)
    print('Full Out-Of-Fold score : %9.6f' 
          % (mean_squared_error(np.log1p(y['target']), oof_preds) ** .5))

    return oof_preds, sub_preds
    
def main():
    # Get the data
    data, test = get_data()
    
    # Get target and ids 
    y = data[['ID', 'target']].copy()
    del data['target'], data['ID']
    sub = test[['ID']].copy()
    del test['ID']
    
    # Free some memory
    gc.collect()

    # Add features
    data, test = add_statistics(data, test)
    
    # Predict test target
    oof_preds, sub_preds = fit_predict(data, y, test)
    
    # Store predictions
    y['predictions'] = np.expm1(oof_preds)
    y[['ID', 'target', 'predictions']].to_csv('reduced_set_oof.csv', index=False)
    sub['target'] = np.expm1(sub_preds)
    sub[['ID', 'target']].to_csv('reduced_set_submission.csv', index=False)

if __name__ == '__main__':
    main()