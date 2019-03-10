import gc
import os
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')

#logger
def get_logger():
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger
    
logger = get_logger()

def read_data(nrows=None):
    logger.info('Input data')
    train_df = pd.read_csv('../input/train.csv',nrows=nrows)
    test_df = pd.read_csv('../input/test.csv')
    return train_df, test_df

def process_data(train_df, test_df):
    logger.info('Features engineering')
    idx = [c for c in train_df.columns if c not in ['ID_code', 'target']]
    for df in [test_df, train_df]:
        for feat in idx:
            df['r2_'+feat] = np.round(df[feat], 2)
            df['r2_'+feat] = np.round(df[feat], 2)
        df['sum'] = df[idx].sum(axis=1)  
        df['min'] = df[idx].min(axis=1)
        df['max'] = df[idx].max(axis=1)
        df['mean'] = df[idx].mean(axis=1)
        df['std'] = df[idx].std(axis=1)
        df['skew'] = df[idx].skew(axis=1)
        df['kurt'] = df[idx].kurtosis(axis=1)
        df['med'] = df[idx].median(axis=1)
    print('Train and test shape:',train_df.shape, test_df.shape)
    return train_df, test_df

def run_model(train_df, test_df):
    logger.info('Prepare the model')
    features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
    target = train_df['target']
    logger.info('Run model')
    param = {
        'bagging_freq': 5,
        'bagging_fraction': 0.38,
        'boost_from_average':'false',
        'boost': 'gbdt',
        'feature_fraction': 0.045,
        'learning_rate': 0.0095,
        'max_depth': -1,  
        'metric':'auc',
        'min_data_in_leaf': 80,
        'min_sum_hessian_in_leaf': 10.0,
        'num_leaves': 13,
        'num_threads': 8,
        'tree_learner': 'serial',
        'objective': 'binary', 
        'verbosity': 1
    }
    num_round = 1000000
    folds = StratifiedKFold(n_splits=12, shuffle=False, random_state=44000)
    oof = np.zeros(len(train_df))
    predictions = np.zeros(len(test_df))
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
        print("Fold {}".format(fold_))
        trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])
        val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])
        clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3500)
        oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)
        predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits
    print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
    return predictions

def submit(test_df, predictions):
    logger.info('Prepare submission')
    sub = pd.DataFrame({"ID_code": test_df.ID_code.values})
    sub["target"] = predictions
    sub.to_csv("submission.csv", index=False)


def main(nrows=None):
    train_df, test_df = read_data(nrows)
    #train_df, test_df = process_data(train_df, test_df)
    predictions = run_model(train_df, test_df)
    submit(test_df, predictions)
    
    
if __name__ == "__main__":
    main()
    