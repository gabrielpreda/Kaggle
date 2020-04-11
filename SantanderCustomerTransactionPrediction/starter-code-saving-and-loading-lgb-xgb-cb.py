import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import pickle
import os
import gc
gc.enable()


def fit_lgb(X_fit, y_fit, X_val, y_val, counter, lgb_path, name):
    
    model = lgb.LGBMClassifier(max_depth=-1,
                               n_estimators=999999,
                               learning_rate=0.02,
                               colsample_bytree=0.3,
                               num_leaves=2,
                               metric='auc',
                               objective='binary', 
                               n_jobs=-1)
     
    model.fit(X_fit, y_fit, 
              eval_set=[(X_val, y_val)],
              verbose=0, 
              early_stopping_rounds=1000)
                  
    cv_val = model.predict_proba(X_val)[:,1]
    
    #Save LightGBM Model
    save_to = '{}{}_fold{}.txt'.format(lgb_path, name, counter+1)
    model.booster_.save_model(save_to)
    
    return cv_val
    
    
def fit_xgb(X_fit, y_fit, X_val, y_val, counter, xgb_path, name):
    
    model = xgb.XGBClassifier(max_depth=2,
                              n_estimators=999999,
                              colsample_bytree=0.3,
                              learning_rate=0.02,
                              objective='binary:logistic', 
                              n_jobs=-1)
     
    model.fit(X_fit, y_fit, 
              eval_set=[(X_val, y_val)], 
              verbose=0, 
              early_stopping_rounds=1000)
              
    cv_val = model.predict_proba(X_val)[:,1]
    
    #Save XGBoost Model
    save_to = '{}{}_fold{}.dat'.format(xgb_path, name, counter+1)
    pickle.dump(model, open(save_to, "wb"))
    
    return cv_val
    
    
def fit_cb(X_fit, y_fit, X_val, y_val, counter, cb_path, name):
    
    model = cb.CatBoostClassifier(iterations=999999,
                                  max_depth=2,
                                  learning_rate=0.02,
                                  colsample_bylevel=0.03,
                                  objective="Logloss")
                                  
    model.fit(X_fit, y_fit, 
              eval_set=[(X_val, y_val)], 
              verbose=0, early_stopping_rounds=1000)
              
    cv_val = model.predict_proba(X_val)[:,1]
    
    #Save Catboost Model          
    save_to = "{}{}_fold{}.mlmodel".format(cb_path, name, counter+1)
    model.save_model(save_to, format="coreml", 
                     export_parameters={'prediction_type': 'probability'})
                     
    return cv_val


def train_stage(df_path, lgb_path, xgb_path, cb_path):
    
    print('Load Train Data.')
    df = pd.read_csv(df_path)
    print('\nShape of Train Data: {}'.format(df.shape))
    
    y_df = np.array(df['target'])                        
    df_ids = np.array(df.index)                     
    df.drop(['ID_code', 'target'], axis=1, inplace=True)
    
    lgb_cv_result = np.zeros(df.shape[0])
    xgb_cv_result = np.zeros(df.shape[0])
    cb_cv_result  = np.zeros(df.shape[0])
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    skf.get_n_splits(df_ids, y_df)
    
    print('\nModel Fitting...')
    for counter, ids in enumerate(skf.split(df_ids, y_df)):
        print('\nFold {}'.format(counter+1))
        X_fit, y_fit = df.values[ids[0]], y_df[ids[0]]
        X_val, y_val = df.values[ids[1]], y_df[ids[1]]
    
        print('LigthGBM')
        lgb_cv_result[ids[1]] += fit_lgb(X_fit, y_fit, X_val, y_val, counter, lgb_path, name='lgb')
        print('XGBoost')
        xgb_cv_result[ids[1]] += fit_xgb(X_fit, y_fit, X_val, y_val, counter, xgb_path, name='xgb')
        print('CatBoost')
        cb_cv_result[ids[1]]  += fit_cb(X_fit,  y_fit, X_val, y_val, counter, cb_path,  name='cb')
        
        del X_fit, X_val, y_fit, y_val
        gc.collect()
    
    auc_lgb  = round(roc_auc_score(y_df, lgb_cv_result),4)
    auc_xgb  = round(roc_auc_score(y_df, xgb_cv_result),4)
    auc_cb   = round(roc_auc_score(y_df, cb_cv_result), 4)
    auc_mean = round(roc_auc_score(y_df, (lgb_cv_result+xgb_cv_result+cb_cv_result)/3), 4)
    auc_mean_lgb_cb = round(roc_auc_score(y_df, (lgb_cv_result+cb_cv_result)/2), 4)
    print('\nLightGBM VAL AUC: {}'.format(auc_lgb))
    print('XGBoost  VAL AUC: {}'.format(auc_xgb))
    print('Catboost VAL AUC: {}'.format(auc_cb))
    print('Mean Catboost+LightGBM VAL AUC: {}'.format(auc_mean_lgb_cb))
    print('Mean XGBoost+Catboost+LightGBM, VAL AUC: {}\n'.format(auc_mean))
    
    return 0
    
    
def prediction_stage(df_path, lgb_path, xgb_path, cb_path):
    
    print('Load Test Data.')
    df = pd.read_csv(df_path)
    print('\nShape of Test Data: {}'.format(df.shape))
    
    df.drop(['ID_code'], axis=1, inplace=True)
    
    lgb_models = sorted(os.listdir(lgb_path))
    xgb_models = sorted(os.listdir(xgb_path))
    cb_models  = sorted(os.listdir(cb_path))
    
    lgb_result = np.zeros(df.shape[0])
    xgb_result = np.zeros(df.shape[0])
    cb_result  = np.zeros(df.shape[0])
    
    print('\nMake predictions...\n')
    
    print('With LightGBM...')
    for m_name in lgb_models:
        #Load LightGBM Model
        model = lgb.Booster(model_file='{}{}'.format(lgb_path, m_name))
        lgb_result += model.predict(df.values)
     
    print('With XGBoost...')    
    for m_name in xgb_models:
        #Load Catboost Model
        model = pickle.load(open('{}{}'.format(xgb_path, m_name), "rb"))
        xgb_result += model.predict(df.values)
    
    print('With CatBoost...')        
    for m_name in cb_models:
        #Load Catboost Model
        model = cb.CatBoostClassifier()
        model = model.load_model('{}{}'.format(cb_path, m_name), format = 'coreml')
        cb_result += model.predict(df.values, prediction_type='Probability')[:,1]
    
    lgb_result /= len(lgb_models)
    xgb_result /= len(xgb_models)
    cb_result  /= len(cb_models)
    
    submission = pd.read_csv('../input/sample_submission.csv')
    submission['target'] = (lgb_result+xgb_result+cb_result)/3
    submission.to_csv('xgb_lgb_cb_starter_submission.csv', index=False)
    submission['target'] = (lgb_result+cb_result)/2
    submission.to_csv('lgb_cb_starter_submission.csv', index=False)
    submission['target'] = xgb_result
    submission.to_csv('xgb_starter_submission.csv', index=False)
    submission['target'] = lgb_result
    submission.to_csv('lgb_starter_submission.csv', index=False)
    submission['target'] = cb_result
    submission.to_csv('cb_starter_submission.csv', index=False)
    
    return 0
    
    
if __name__ == '__main__':
    
    train_path = '../input/train.csv'
    test_path  = '../input/test.csv'
    
    lgb_path = './lgb_models_stack/'
    xgb_path = './xgb_models_stack/'
    cb_path  = './cb_models_stack/'

    #Create dir for models
    os.mkdir(lgb_path)
    os.mkdir(xgb_path)
    os.mkdir(cb_path)

    print('Train Stage.\n')
    train_stage(train_path, lgb_path, xgb_path, cb_path)
    
    print('Prediction Stage.\n')
    prediction_stage(test_path, lgb_path, xgb_path, cb_path)
    
    print('\nDone.')