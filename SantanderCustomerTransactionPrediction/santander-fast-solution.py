import gc
import os
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
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
logger.info('Input data')
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

logger.info('Features engineering')
idx = [c for c in train_df.columns if c not in ['ID_code', 'target']]
for df in [test_df, train_df]:
    df['sum'] = df[idx].sum(axis=1)  
    df['min'] = df[idx].min(axis=1)
    df['max'] = df[idx].max(axis=1)
    df['mean'] = df[idx].mean(axis=1)
    df['std'] = df[idx].std(axis=1)
    df['skew'] = df[idx].skew(axis=1)
    df['kurt'] = df[idx].kurtosis(axis=1)
    df['med'] = df[idx].median(axis=1)

logger.info('Prepare the model')
features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
target = train_df['target']
logger.info('Run model')
params = {'tree_method': 'gpu_hist',
          'max_depth': 7,
          'alpha': 0.1,
          'gamma': 0.3,
          'subsample': 0.65,
          'scale_pos_weight': 1,
          'learning_rate': 0.03, 
          'silent': 1, 
          'objective':'binary:logistic', 
          'eval_metric': 'auc',
          'n_gpus': 1}

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
    print("Fold {}".format(fold_))

    X_train = xgb.DMatrix(train_df.iloc[trn_idx][features].values, target.iloc[trn_idx].values)
    X_valid = xgb.DMatrix(train_df.iloc[val_idx][features].values, target.iloc[val_idx].values)   
    clf = xgb.train(params, X_train, 2000, evals=[(X_train, "train"), (X_valid, "eval")],
                early_stopping_rounds=50, verbose_eval=100)
    oof[val_idx] = clf.predict(X_valid) 
    predictions += clf.predict(xgb.DMatrix(test_df[features].values)) / folds.n_splits


print("CV score: {:<8.5f}".format(roc_auc_score(target.values, oof)))


logger.info('Prepare submission')
sub = pd.DataFrame({"ID_code": test_df.ID_code.values})
sub["target"] = predictions
sub.to_csv("submission.csv", index=False)