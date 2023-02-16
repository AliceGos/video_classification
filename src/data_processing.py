########################
#### Data Processing ###
########################

#%% Load libraries
import os
from modules_dataprep import get_accgyro, saving_json_as_dict,replacenth, create_newidx,get_exploinfo
from modules_viz import plot_columns
import pandas as pd
import numpy as np
from random import sample
from sklearn.model_selection import StratifiedShuffleSplit
import re
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

#%% Fetch files list
path = '/home/I0259079/workdir/surf_dataset/'
path_to_json   = path+'clean_json_surf/'
json_files     = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
json_longboard = list(filter(lambda x: 'longboard' in x, json_files))
json_surfboard = [x for x in json_files if x not in json_longboard]

#%% Run once to save json as dictionnary with no key duplicates
# newlist = [x[:-5] for x in json_surfboard]
# dictnames = [s + '_dict' for s in newlist]
# saving_json_as_dict(
#     path_to_json=path_to_json,
#     files_list=json_surfboard,
#     newnames_list=dictnames
# )

# newlist = [x[:-5] for x in json_longboard[:]]
# dictnames = [s + '_dict' for s in newlist]
# saving_json_as_dict(
#     path_to_json=path_to_json,
#     files_list=json_longboard[:],
#     newnames_list=dictnames
# )

#%% Remove outlier
json_surfboard.remove('GX010799_no.MP4.json')

#%% Plots of global information
surf_type = json_longboard

newlist   = [x[:-5] for x in surf_type]
dictnames = [s + '_dict' for s in newlist]
explo_df  = get_exploinfo(path_to_json=path_to_json,dictnames=dictnames)
                                                                
values, counts = np.unique(explo_df['gopro_model'], return_counts=True)
print(values, counts)

explo_df=explo_df.replace('fall','no')
print(explo_df.describe())

plt.figure()
sns.boxplot(data=explo_df, x='takeoff', y='metadata_len')
plt.title('Seconds of metadata \n longboard')
plt.xlabel('Take off Yes/No')
plt.ylabel('Seconds')
plt.show()

plt.figure()
sns.boxplot(data=explo_df, x='takeoff', y='freq_accl')
plt.title('Frequency of accelerometer/gyroscope \n longboard')
plt.xlabel('Take off Yes/No')
plt.ylabel('Hz')
plt.show()

#%% Create dataframe containing accelerometer and gyroscope data
accgyro_surf = get_accgyro(
    path_to_json = path_to_json,
    files_list   = json_surfboard
    )
accgyro_longboard = get_accgyro(
    path_to_json = path_to_json,
    files_list   = json_longboard
    )

#%% Explore features for a few videos
json_surfboard_sub = [x.split('_')[0] for x in sample(json_surfboard,3)]
for name in json_surfboard_sub:
    plot_columns(
        full_dataset = accgyro_surf, 
        video_id     = name
    )

#%% Histograms of features
cols = list(accgyro_surf.columns[:6])
for col in cols:
    plt.figure()
    sns.histplot(accgyro_surf,x=col,hue='video_id')
    plt.title('Surf video distribution')
    plt.show()
    
for col in cols:
    plt.figure()
    sns.histplot(accgyro_longboard,x=col,hue='video_id')
    plt.title('Longboard video distribution')
    plt.show()

#%% Normalize by video
cols_to_scale = list(accgyro_surf.columns[:6])
accgyro_leave_col = accgyro_surf.drop(columns=cols_to_scale)
accgyro_kept_col = accgyro_surf[cols_to_scale+['video_id']]
tot = pd.DataFrame()
for id in accgyro_kept_col.video_id.unique():
    tp = accgyro_kept_col[accgyro_kept_col['video_id']==id]
    tp = tp[cols_to_scale].apply(lambda x: (x - x.mean()) / x.std())
    tot = pd.concat((tot,tp),axis=0)

accgyro_surf_norm = pd.concat((tot, accgyro_leave_col.reset_index()), axis=1)

cols_to_scale = list(accgyro_longboard.columns[:6])
accgyro_leave_col = accgyro_longboard.drop(columns=cols_to_scale)
accgyro_kept_col = accgyro_longboard[cols_to_scale+['video_id']]
tot = pd.DataFrame()
for id in accgyro_kept_col.video_id.unique():
    tp = accgyro_kept_col[accgyro_kept_col['video_id']==id]
    tp = tp[cols_to_scale].apply(lambda x: (x - x.mean()) / x.std())
    tot = pd.concat((tot,tp),axis=0)

accgyro_longboard_norm = pd.concat((tot, accgyro_leave_col.reset_index()), axis=1)

#%% Histograms of features after normalization
cols = list(accgyro_surf_norm.columns[:6])
for col in cols:
    plt.figure()
    sns.histplot(accgyro_surf_norm,x=col,hue='video_id')
    plt.title('Surf video distribution')
    plt.show()
    
for col in cols:
    plt.figure()
    sns.histplot(accgyro_longboard_norm,x=col,hue='video_id')
    plt.title('Longboard video distribution')
    plt.show()

############################
#%% Create train and test datasets when using only surf videos
video_label = accgyro_surf_norm[['video_id', 'label']].value_counts().reset_index(name='count')
X = video_label['video_id']
y = video_label['label']
n_splits = 1  
sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=0)
for train_index, test_index in sss.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

train = accgyro_surf_norm[accgyro_surf_norm['video_id'].isin(X_train)]
test  = accgyro_surf_norm[accgyro_surf_norm['video_id'].isin(X_test)]

#oversample = SMOTE()
#train, y_train = oversample.fit_resample(train, y_train)

create_newidx(test)

#%% Look at nb of yes/no in train and test
nb_yes_train = len(list(train.loc[train['label'] == 1, 'video_id'].unique()))
nb_no_train  = len(list(train.loc[train['label'] == 0, 'video_id'].unique()))
nb_yes_test  = len(list(test.loc[test['label'] == 1, 'video_id'].unique()))
nb_no_test   = len(list(test.loc[test['label'] == 0, 'video_id'].unique()))
print(
    'Number of yes in train set: ', str(nb_yes_train),'\n',
    'Number of no in train set: ', str(nb_no_train),'\n',
    'Number of yes in test set: ', str(nb_yes_test),'\n',
    'Number of no in test set: ', str(nb_no_test),'\n',
)

#%% Create train, validation and test datasets when using both surf and longboard videos
accgyro = pd.concat((accgyro_surf_norm,accgyro_longboard_norm),axis=0)
video_label = accgyro[['video_id', 'label']].value_counts().reset_index(name='count')
X = video_label['video_id']
y = video_label['label']
n_splits = 1  # We only want a single split in this case
sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=0)
for train_index, test_index in sss.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train_all, y_test_all = y[train_index], y[test_index]

train_all = accgyro[accgyro['video_id'].isin(X_train)]
test_all  = accgyro[accgyro['video_id'].isin(X_test)]

create_newidx(test_all)

#%% Look at nb of yes/no in train and test
nb_yes_train = len(list(train_all.loc[train_all['label'] == 1, 'video_id'].unique()))
nb_no_train  = len(list(train_all.loc[train_all['label'] == 0, 'video_id'].unique()))
nb_yes_test  = len(list(test_all.loc[test_all['label'] == 1, 'video_id'].unique()))
nb_no_test   = len(list(test_all.loc[test_all['label'] == 0, 'video_id'].unique()))
print(
    'Number of yes in train set: ', str(nb_yes_train),'\n',
    'Number of no in train set: ', str(nb_no_train),'\n',
    'Number of yes in test set: ', str(nb_yes_test),'\n',
    'Number of no in test set: ', str(nb_no_test),'\n',
)

# %% Save files
train.to_csv(path+'train.csv',index=0)
test.to_csv(path+'test.csv',index=0)
y_test.to_csv(path+'y_test.csv',index=0)
train_all.to_csv(path+'train_all.csv',index=0)
test_all.to_csv(path+'test_all.csv',index=0)
y_test_all.to_csv(path+'y_test_all.csv',index=0)

# %%
