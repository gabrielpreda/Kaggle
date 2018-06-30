#------------------------------------------------------------------------------
NROWS_TRAIN=184903891         #dimmension of the train set
NCHUNK_TRAIN=75000000         #length of chunk of data used for training, from total train set
MAX_TRAIN=75000000            #max length of train data (substract from NROWS_TRAIN to get the start position for training set)
NROWS_VALIDATION=2500000      #size of the validation set
ENV_RUN='local'               #environment where the kernel is run
PRESET_D = 2 ** 26
PRESET_DM = 3000000000

if ENV_RUN=='local':
	inpath = '../input/'
	suffix = ''
	outpath = ''
	savepath = ''    
elif ENV_RUN=='aws':
	inpath = '../input/'
	suffix = '.zip'
	outpath = '../sub/'
	savepath = '../data/'
#------------------------------------------------------------------------------
import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt
import os
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def show_max_clean(df,gp,agg_name,agg_type,show_max):
#------------------------------------------------------------------------------    
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )
    
#------------------------------------------------------------------------------
def perform_count( df, group_cols, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
#------------------------------------------------------------------------------
    if show_agg:
        print( "Aggregating by ", group_cols , '...' )
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    return (show_max_clean(df,gp,agg_name,agg_type,show_max))

#------------------------------------------------------------------------------
def perform_countuniq( df, group_cols, counted, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
#------------------------------------------------------------------------------    
    if show_agg:
        print( "Counting unique ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].nunique().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    return (show_max_clean(df,gp,agg_name,agg_type,show_max))

#------------------------------------------------------------------------------    
def perform_cumcount( df, group_cols, counted, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
#------------------------------------------------------------------------------    
    if show_agg:
        print( "Cumulative count by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].cumcount()
    df[agg_name]=gp.values
    return (show_max_clean(df,gp,agg_name,agg_type,show_max))

#------------------------------------------------------------------------------
def perform_mean( df, group_cols, counted, agg_name, agg_type='float32', show_max=False, show_agg=True ):
#------------------------------------------------------------------------------    
    if show_agg:
        print( "Calculating mean of ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].mean().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    return (show_max_clean(df,gp,agg_name,agg_type,show_max))

#------------------------------------------------------------------------------
def perform_var( df, group_cols, counted, agg_name, agg_type='float32', show_max=False, show_agg=True ):
#------------------------------------------------------------------------------    
    if show_agg:
        print( "Calculating variance of ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].var().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    return (show_max_clean(df,gp,agg_name,agg_type,show_max))

debug=0 
if debug:
    print('*** debug parameter set: this is a test run for debugging purposes ***')


#------------------------------------------------------------------------------
def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                 feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None):
#------------------------------------------------------------------------------    
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
        'learning_rate': 0.2,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 4,
        'verbose': 0,
        'metric':metrics
    }

    lgb_params.update(params)

    print("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

    evals_results = {}

    bst1 = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgtrain, xgvalid], 
                     valid_names=['train','valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10, 
                     feval=feval)

    print("\nModel Report")
    print("bst1.best_iteration: ", bst1.best_iteration)
    print(metrics+":", evals_results['valid'][metrics][bst1.best_iteration-1])

    return (bst1,bst1.best_iteration)

#------------------------------------------------------------------------------
def perform_analysis(idx_from,idx_to,fileno):
#------------------------------------------------------------------------------    
    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint16',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32',
            }

    print('loading train data...',idx_from,idx_to)
    train_df = pd.read_csv(inpath+"train.csv", parse_dates=['click_time'], 
        skiprows=range(1,idx_from), nrows=idx_to-idx_from, dtype=dtypes, 
        usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])

    print('loading test data...')
    if debug:
        test_df = pd.read_csv(inpath+"test.csv", nrows=100000, 
        parse_dates=['click_time'], dtype=dtypes, 
        usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
    else:
        test_df = pd.read_csv(inpath+"test.csv", parse_dates=['click_time'], 
        dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])

    len_train = len(train_df)
    train_df=train_df.append(test_df)
    
    del test_df
    gc.collect()
    
    print('Extracting new features...')
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    
    gc.collect()
    train_df = perform_countuniq( train_df, ['ip'], 'channel', 'X0', 'uint8', show_max=True ); gc.collect()
    train_df = perform_cumcount( train_df, ['ip', 'device', 'os'], 'app', 'X1', show_max=True ); gc.collect()
    train_df = perform_countuniq( train_df, ['ip', 'day'], 'hour', 'X2', 'uint8', show_max=True ); gc.collect()
    train_df = perform_countuniq( train_df, ['ip'], 'app', 'X3', 'uint8', show_max=True ); gc.collect()
    train_df = perform_countuniq( train_df, ['ip', 'app'], 'os', 'X4', 'uint8', show_max=True ); gc.collect()
    train_df = perform_countuniq( train_df, ['ip'], 'device', 'X5', 'uint16', show_max=True ); gc.collect()
    train_df = perform_countuniq( train_df, ['app'], 'channel', 'X6', show_max=True ); gc.collect()
    train_df = perform_cumcount( train_df, ['ip'], 'os', 'X7', show_max=True ); gc.collect()
    train_df = perform_countuniq( train_df, ['ip', 'device', 'os'], 'app', 'X8', show_max=True ); gc.collect()
    train_df = perform_count( train_df, ['ip', 'day', 'hour'], 'ip_tcount', show_max=True ); gc.collect()
    train_df = perform_count( train_df, ['ip', 'app'], 'ip_app_count', show_max=True ); gc.collect()
    train_df = perform_count( train_df, ['ip', 'app', 'os'], 'ip_app_os_count', 'uint16', show_max=True ); gc.collect()
    train_df = perform_var( train_df, ['ip', 'day', 'channel'], 'hour', 'ip_tchan_count', show_max=True ); gc.collect()
    train_df = perform_var( train_df, ['ip', 'app', 'os'], 'hour', 'ip_app_os_var', show_max=True ); gc.collect()
    train_df = perform_var( train_df, ['ip', 'app', 'channel'], 'day', 'ip_app_channel_var_day', show_max=True ); gc.collect()
    train_df = perform_mean( train_df, ['ip', 'app', 'channel'], 'hour', 'ip_app_channel_mean_hour', show_max=True ); gc.collect()

    print('doing nextClick')
    predictors=[]
    
    new_feature = 'nextClick'
    filename='nextClick_%d_%d.csv'%(idx_from,idx_to)

    #if os.path.exists(filename):
    if (0):
        print('loading from save file')
        QQ=pd.read_csv(filename).values
    else:
        D=PRESET_D
        train_df['category'] = (train_df['ip'].astype(str) + "_" + train_df['app'].astype(str) + "_" + train_df['device'].astype(str) \
            + "_" + train_df['os'].astype(str)).apply(hash) % D
        click_buffer= np.full(D, PRESET_DM, dtype=np.uint32)

        train_df['epochtime']= train_df['click_time'].astype(np.int64) // 10 ** 9
        next_clicks= []
        for category, t in zip(reversed(train_df['category'].values), reversed(train_df['epochtime'].values)):
            next_clicks.append(click_buffer[category]-t)
            click_buffer[category]= t
        del(click_buffer)
        QQ= list(reversed(next_clicks))

        if not debug:
            print('saving')
            pd.DataFrame(QQ).to_csv(filename,index=False)
            
    train_df.drop(['epochtime','category','click_time'], axis=1, inplace=True)
    #additional drop columns(lee importance)
    train_df.drop(['day','ip_tchan_count','X7'],axis=1, inplace=True)

    train_df[new_feature] = pd.Series(QQ).astype('float32')
    predictors.append(new_feature)

    train_df[new_feature+'_shift'] = train_df[new_feature].shift(+1).values
    predictors.append(new_feature+'_shift')
    
    del QQ
    gc.collect()

    print("vars and data type: ")
    train_df.info()
    train_df['ip_tcount'] = train_df['ip_tcount'].astype('uint16')
    train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
    train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')

    target = 'is_attributed'
# =============================================================================
#     predictors.extend(['app','device','os', 'channel', 'hour', 'day', 
#                   'ip_tcount', 'ip_tchan_count', 'ip_app_count',
#                   'ip_app_os_count', 'ip_app_os_var',
#                   'ip_app_channel_var_day','ip_app_channel_mean_hour',
#                   'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8'])
#     categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']
#     
# =============================================================================
    predictors.extend(['app','device','os', 'channel', 'hour', 
                  'ip_tcount',  'ip_app_count',
                  'ip_app_os_count', 'ip_app_os_var',
                  'ip_app_channel_var_day','ip_app_channel_mean_hour',
                  'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8'])
    categorical = ['app', 'device', 'os', 'channel', 'hour']
    
    
    print('predictors',predictors)

    test_df = train_df[len_train:]
    val_df = train_df[(len_train-val_size):len_train]
    train_df = train_df[:(len_train-val_size)]

    print("train size: ", len(train_df))
    print("valid size: ", len(val_df))
    print("test size : ", len(test_df))

    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')

    gc.collect()

    print("Training...")
    start_time = time.time()

    params = {
        'learning_rate': 0.20,
        #'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 7,  # 2^max_depth - 1
        'max_depth': 3,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight':200 # because training data is extremely unbalanced 
    }
    (bst,best_iteration) = lgb_modelfit_nocv(params, 
                            train_df, 
                            val_df, 
                            predictors, 
                            target, 
                            objective='binary', 
                            metrics='auc',
                            early_stopping_rounds=30, 
                            verbose_eval=True, 
                            num_boost_round=1000, 
                            categorical_features=categorical)

    print('[{}]: model training time'.format(time.time() - start_time))
    del train_df
    del val_df
    gc.collect()

    if ENV_RUN!='aws':
        print('Plot feature importances...')
        ax = lgb.plot_importance(bst, max_num_features=100)
        plt.show()

    print("Predicting...")
    sub['is_attributed'] = bst.predict(test_df[predictors],num_iteration=best_iteration)
    if not debug:
        print("writing...")
        sub.to_csv('sub_it%d.csv'%(fileno),index=False,float_format='%.9f')
    print("done...")
    return sub
#------------------------------------------------------------------------------


nrows=NROWS_TRAIN-1
nchunk= NCHUNK_TRAIN 
val_size=NROWS_VALIDATION
idx_from=nrows-MAX_TRAIN

if debug:
    idx_from=0
    nchunk=100000
    val_size=10000

idx_to=idx_from+nchunk

sub= perform_analysis(idx_from,idx_to,0)
