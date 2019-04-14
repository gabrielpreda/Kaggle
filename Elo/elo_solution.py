#packages
import gc
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
#settings
warnings.filterwarnings('ignore')
np.random.seed(2019)
#logger
def get_logger():
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger

IS_LOCAL = True

if(IS_LOCAL):
    PATH="../input/elo-merchant-category-recommendation/"
else:
    PATH="../input/"
 
logger = get_logger()
#load data
logger.info('Start read data')
train_df = pd.read_csv(PATH+'train.csv')
test_df = pd.read_csv(PATH+'test.csv')
historical_trans_df = pd.read_csv(PATH+'historical_transactions.csv')
new_merchant_trans_df = pd.read_csv(PATH+'new_merchant_transactions.csv')

#process NAs
logger.info('Start processing NAs')
#process NA2 for transactions
for df in [historical_trans_df, new_merchant_trans_df]:
    df['category_2'].fillna(1.0,inplace=True)
    df['category_3'].fillna('A',inplace=True)
    df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
#define function for aggregation
def create_new_columns(name,aggs):
    return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]
   
logger.info('process historical and new merchant datasets')
for df in [historical_trans_df, new_merchant_trans_df]:
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['year'] = df['purchase_date'].dt.year
    df['weekofyear'] = df['purchase_date'].dt.weekofyear
    df['month'] = df['purchase_date'].dt.month
    df['dayofweek'] = df['purchase_date'].dt.dayofweek
    df['weekend'] = (df.purchase_date.dt.weekday >=5).astype(int)
    df['hour'] = df['purchase_date'].dt.hour
    df['authorized_flag'] = df['authorized_flag'].map({'Y':1, 'N':0})
    df['category_1'] = df['category_1'].map({'Y':1, 'N':0}) 
    df['category_3'] = df['category_3'].map({'A':0, 'B':1, 'C':2}) 
    df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days)//30
    df['month_diff'] += df['month_lag']


#define aggregations with historical_trans_df
logger.info('Aggregate historical trans')
aggs = {}

for col in ['subsector_id','merchant_id','merchant_category_id']:
    aggs[col] = ['nunique']
for col in ['month', 'hour', 'weekofyear', 'dayofweek']:
    aggs[col] = ['nunique', 'mean', 'min', 'max']
    

aggs['purchase_amount'] = ['sum','max','min','mean','var', 'std']
aggs['installments'] = ['sum','max','min','mean','var', 'std']
aggs['purchase_date'] = ['max','min', 'nunique']
aggs['month_lag'] = ['max','min','mean','var']
aggs['month_diff'] = ['mean', 'min', 'max', 'var']
aggs['authorized_flag'] = ['sum', 'mean', 'nunique']
aggs['weekend'] = ['sum', 'mean', 'nunique']
aggs['year'] = ['nunique', 'mean']
aggs['category_1'] = ['sum', 'mean', 'min', 'max', 'nunique']
aggs['category_2'] = ['sum', 'mean', 'min', 'nunique']
aggs['category_3'] = ['sum', 'mean', 'min', 'nunique']
aggs['card_id'] = ['size', 'count', 'nunique']

for col in ['category_1','category_2','category_3', 'month_lag','month_diff']:
    historical_trans_df[col+'_mean'] = historical_trans_df.groupby([col])['purchase_amount'].transform('mean')
    historical_trans_df[col+'_min'] = historical_trans_df.groupby([col])['purchase_amount'].transform('min')
    historical_trans_df[col+'_max'] = historical_trans_df.groupby([col])['purchase_amount'].transform('max')
    historical_trans_df[col+'_sum'] = historical_trans_df.groupby([col])['purchase_amount'].transform('sum')
    historical_trans_df[col+'_std'] = historical_trans_df.groupby([col])['purchase_amount'].transform('std')
    aggs[col+'_mean'] = ['mean']
    aggs[col+'_min'] = ['min']
    aggs[col+'_max'] = ['max']
    aggs[col+'_sum'] = ['sum']
    

new_columns = create_new_columns('hist',aggs)
historical_trans_group_df = historical_trans_df.groupby('card_id').agg(aggs)
historical_trans_group_df.columns = new_columns
historical_trans_group_df.reset_index(drop=False,inplace=True)
historical_trans_group_df['hist_purchase_date_diff'] = (historical_trans_group_df['hist_purchase_date_max'] - historical_trans_group_df['hist_purchase_date_min']).dt.days
historical_trans_group_df['hist_purchase_date_average'] = historical_trans_group_df['hist_purchase_date_diff']/historical_trans_group_df['hist_card_id_size']
historical_trans_group_df['hist_purchase_date_uptonow'] = (datetime.datetime.today() - historical_trans_group_df['hist_purchase_date_max']).dt.days
historical_trans_group_df['hist_purchase_date_uptomin'] = (datetime.datetime.today() - historical_trans_group_df['hist_purchase_date_min']).dt.days


#merge with train, test
train_df = train_df.merge(historical_trans_group_df,on='card_id',how='left')
test_df = test_df.merge(historical_trans_group_df,on='card_id',how='left')

#cleanup memory
del historical_trans_group_df; gc.collect()

#define aggregations with new_merchant_trans_df 
logger.info('Aggregate new merchant trans')
aggs = {}
for col in ['subsector_id','merchant_id','merchant_category_id']:
    aggs[col] = ['nunique']
for col in ['month', 'hour', 'weekofyear', 'dayofweek']:
    aggs[col] = ['nunique', 'mean', 'min', 'max']

aggs['purchase_amount'] = ['sum','max','min','mean','var']
aggs['installments'] = ['sum','max','min','mean','var']
aggs['purchase_date'] = ['max','min', 'nunique']
aggs['month_lag'] = ['max','min','mean','var']
aggs['month_diff'] = ['mean', 'max', 'min', 'var']
aggs['weekend'] = ['sum', 'mean', 'nunique']
aggs['year'] = ['nunique', 'mean']
aggs['category_1'] = ['sum', 'mean', 'min', 'nunique']
aggs['category_2'] = ['sum', 'mean', 'min', 'nunique']
aggs['category_3'] = ['sum', 'mean', 'min', 'nunique']
aggs['card_id'] = ['size', 'count',  'nunique']


for col in ['category_1','category_2','category_3', 'month_lag', 'month_diff']:
    new_merchant_trans_df[col+'_mean'] = new_merchant_trans_df.groupby([col])['purchase_amount'].transform('mean')
    new_merchant_trans_df[col+'_min'] = new_merchant_trans_df.groupby([col])['purchase_amount'].transform('min')
    new_merchant_trans_df[col+'_max'] = new_merchant_trans_df.groupby([col])['purchase_amount'].transform('max')
    new_merchant_trans_df[col+'_sum'] = new_merchant_trans_df.groupby([col])['purchase_amount'].transform('sum')
    aggs[col+'_mean'] = ['mean']
    aggs[col+'_min'] = ['min']
    aggs[col+'_max'] = ['max']
    aggs[col+'_sum'] = ['sum']

    
new_columns = create_new_columns('new_hist',aggs)
new_merchant_trans_group_df = new_merchant_trans_df.groupby('card_id').agg(aggs)
new_merchant_trans_group_df.columns = new_columns
new_merchant_trans_group_df.reset_index(drop=False,inplace=True)
new_merchant_trans_group_df['new_hist_purchase_date_diff'] = (new_merchant_trans_group_df['new_hist_purchase_date_max'] - new_merchant_trans_group_df['new_hist_purchase_date_min']).dt.days
new_merchant_trans_group_df['new_hist_purchase_date_average'] = new_merchant_trans_group_df['new_hist_purchase_date_diff']/new_merchant_trans_group_df['new_hist_card_id_size']
new_merchant_trans_group_df['new_hist_purchase_date_uptonow'] = (datetime.datetime.today() - new_merchant_trans_group_df['new_hist_purchase_date_max']).dt.days
new_merchant_trans_group_df['new_hist_purchase_date_uptomin'] = (datetime.datetime.today() - new_merchant_trans_group_df['new_hist_purchase_date_min']).dt.days
#merge with train, test
train_df = train_df.merge(new_merchant_trans_group_df,on='card_id',how='left')
test_df = test_df.merge(new_merchant_trans_group_df,on='card_id',how='left')

#clean-up memory
del new_merchant_trans_group_df; gc.collect()
del historical_trans_df; gc.collect()
del new_merchant_trans_df; gc.collect()
#process train
logger.info('Process train')
train_df['outliers'] = 0
train_df.loc[train_df['target'] < -30, 'outliers'] = 1
outls = train_df['outliers'].value_counts()
strLog = "outliers: {}".format(outls)
print(strLog)
logger.info('Process train and test')
## process both train and test
for df in [train_df, test_df]:
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['dayofweek'] = df['first_active_month'].dt.dayofweek
    df['weekofyear'] = df['first_active_month'].dt.weekofyear
    df['dayofyear'] = df['first_active_month'].dt.dayofyear
    df['quarter'] = df['first_active_month'].dt.quarter
    df['is_month_start'] = df['first_active_month'].dt.is_month_start
    df['month'] = df['first_active_month'].dt.month
    df['elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days
    df['hist_first_buy'] = (df['hist_purchase_date_min'] - df['first_active_month']).dt.days
    df['hist_last_buy'] = (df['hist_purchase_date_max'] - df['first_active_month']).dt.days
    df['new_hist_first_buy'] = (df['new_hist_purchase_date_min'] - df['first_active_month']).dt.days
    df['new_hist_last_buy'] = (df['new_hist_purchase_date_max'] - df['first_active_month']).dt.days
    for f in ['hist_purchase_date_max','hist_purchase_date_min','new_hist_purchase_date_max',\
                     'new_hist_purchase_date_min']:
        df[f] = df[f].astype(np.int64) * 1e-9
    df['card_id_total'] = df['new_hist_card_id_size']+df['hist_card_id_size']
    df['card_id_cnt_total'] = df['new_hist_card_id_count']+df['hist_card_id_count']
    df['card_id_cnt_nunique'] = df['new_hist_card_id_nunique']+df['hist_card_id_nunique']
    df['purchase_amount_total'] = df['new_hist_purchase_amount_sum']+df['hist_purchase_amount_sum']
    df['purchase_amount_mean'] = df['new_hist_purchase_amount_mean']+df['hist_purchase_amount_mean']
    df['purchase_amount_max'] = df['new_hist_purchase_amount_max']+df['hist_purchase_amount_max']

for f in ['feature_1','feature_2','feature_3']:
    order_label = train_df.groupby([f])['outliers'].mean()
    train_df[f] = train_df[f].map(order_label)
    test_df[f] = test_df[f].map(order_label)
##
train_columns = [c for c in train_df.columns if c not in ['card_id', 'first_active_month','target','outliers']]
target = train_df['target']
del train_df['target']

#model
##model params
logger.info('Prepare model')
param = {'num_leaves': 51,
         'min_data_in_leaf': 35, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.008,
         "boosting": "gbdt",
         "feature_fraction": 0.75,
         "bagging_freq": 1,
         "bagging_fraction": 0.82 ,
         "bagging_seed": 42,
         "metric": 'rmse',
         "lambda_l1": 0.11,
         "verbosity": -1,
         "nthread": 4,
         "random_state": 2019}
#prepare fit model with cross-validation
folds = StratifiedKFold(n_splits=9, shuffle=True, random_state=2019)
oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
feature_importance_df = pd.DataFrame()
#run model
logger.info('Start running model')
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df,train_df['outliers'].values)):
    strLog = "fold {}".format(fold_)
    print(strLog)
    trn_data = lgb.Dataset(train_df.iloc[trn_idx][train_columns], label=target.iloc[trn_idx])#, categorical_feature=categorical_feats)
    val_data = lgb.Dataset(train_df.iloc[val_idx][train_columns], label=target.iloc[val_idx])#, categorical_feature=categorical_feats)

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 150)
    oof[val_idx] = clf.predict(train_df.iloc[val_idx][train_columns], num_iteration=clf.best_iteration)
    #feature importance
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = train_columns
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    #predictions
    predictions += clf.predict(test_df[train_columns], num_iteration=clf.best_iteration) / folds.n_splits
    logger.info(strLog)
    
strRMSE = "RMSE: {}".format(np.sqrt(mean_squared_error(oof, target)))
print(strRMSE)

##plot the feature importance
logger.info("Feature importance plot")
cols = (feature_importance_df[["Feature", "importance"]]
        .groupby("Feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

plt.figure(figsize=(14,26))
sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))
plt.title('LightGBM Features (averaged over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')
##submission
logger.info("Prepare submission")
sub_df = pd.DataFrame({"card_id":test_df["card_id"].values})
sub_df["target"] = predictions
sub_df.to_csv("submission.csv", index=False)