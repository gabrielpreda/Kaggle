import glob
import pandas as pd
import numpy as np

# USING WEIGHTED AVERAGE RANK METHOD
# Plese refer this discussion for more detials - https://www.kaggle.com/c/home-credit-default-risk/discussion/60934

data = {}

for path in glob.glob("../input/homecredt/*.csv", recursive=True):
    data[path[19:-4]] = pd.read_csv(path)

ranks = pd.DataFrame(columns=data.keys())
for key in data.keys():
    print(key)
    ranks[key] = data[key].TARGET.rank(method='min')

ranks['Average'] = ranks.mean(axis=1)
ranks['Scaled Rank'] = (ranks['Average'] - ranks['Average'].min()) / (ranks['Average'].max() - ranks['Average'].min())
ranks.corr()[:1]

weights = [0.0, 0.3, 0.0, 0.0, 0.0, 0.1, 0.6]
ranks['Score'] = ranks[['stack3_diff_data', 'tidy_xgb_0.78847', 'submission', 'submission (1)',
       'hybridII', 'submission_kernel02', 'submission_2018082903']].mul(weights).sum(1) / ranks.shape[0]
submission_lb = pd.read_csv("../input/homecredt/submission.csv")
submission_lb['TARGET'] = ranks['Score']
submission_lb.to_csv("WEIGHT_AVERAGE_RANK.csv", index=None)
submission_lb.head()
## WARNING THIS METHOD IS PRONE TO OVERFITTING..