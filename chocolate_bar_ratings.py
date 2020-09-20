
import os, math, copy, itertools, matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.pylab as pylab

from scipy.sparse import hstack
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.decomposition import PCA, TruncatedSVD, SparsePCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix, log_loss
from category_encoders import TargetEncoder, WOEEncoder, HashingEncoder, OneHotEncoder

from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier

# SMALL_SIZE = 8
# MEDIUM_SIZE = 10
# BIGGER_SIZE = 12

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rcParams.update({'font.size': 7})
dpi_set = 1200

# Read the file into pandas dataframe

main_path = '/media/ashutosh/Computer Vision/Predictive_Maintenance/chocolate-bar-ratings'
#main_path = 'E:\Predictive_Maintenance\chocolate-bar-ratings'
main_data = pd.read_csv(main_path+"//flavors_of_cacao.csv")

features_list = list(main_data.columns)

print(features_list)

#### Plot Ratings
# plot_df = main_data['Rating'].value_counts(normalize=True, sort=True, ascending=False, bins=None, dropna=False)*100
# plot_df_data = list(plot_df)
# plot_df_uniques = list(plot_df.keys())
# plt.figure(dpi=dpi_set,figsize=(4,3))
# plt.xlabel('Ratings')
# plt.ylabel('Percent [%]')
# plt.bar(plot_df_uniques,height=plot_df_data, width=0.17,linewidth=0,edgecolor='w')
# plt.grid(b=True, which='major', axis='both', linestyle=':', linewidth=0.5, alpha=1)
# plt.show()

# #### Plot Review Date
# plot_df = main_data['Review Date'].value_counts(normalize=True, sort=True, ascending=False, bins=None, dropna=False)*100
# plot_df_data = list(plot_df)
# plot_df_uniques = list(plot_df.keys())
# plt.figure(dpi=dpi_set,figsize=(4,3))
# plt.xlabel('Review Date')
# plt.ylabel('Percent [%]')
# plt.bar(plot_df_uniques,height=plot_df_data, width=0.3,linewidth=0,edgecolor='w')
# plt.grid(b=True, which='major', axis='both', linestyle=':', linewidth=0.5, alpha=1)
# plt.show()

# #### Plot REF
# plot_df = main_data['REF'].value_counts(normalize=True, sort=True, ascending=False, bins=None, dropna=False)*100
# plot_df_data = list(plot_df)
# plot_df_uniques = list(plot_df.keys())
# plt.figure(dpi=dpi_set,figsize=(4,3))
# plt.xlabel('Batch Reference No.')
# plt.ylabel('Percent [%]')
# plt.scatter(plot_df_uniques,plot_df_data, s=5)
# plt.grid(b=True, which='major', axis='both', linestyle=':', linewidth=0.5, alpha=1)
# plt.show()

# #### Plot Cocoa Percent
# plot_df = main_data['Cocoa Percent'].value_counts(normalize=True, sort=True, ascending=False, bins=None, dropna=False)*100
# plot_df_data = list(plot_df)
# plot_df_uniques = list(plot_df.keys())
# plt.figure(dpi=dpi_set,figsize=(7.875,3.375))
# plt.xticks(rotation=90)
# plt.xlabel('Cocoa Percent [%]')
# plt.ylabel('Percent [%]')
# plt.scatter(plot_df_uniques,plot_df_data, s=5)
# plt.grid(b=True, which='major', axis='both', linestyle=':', linewidth=0.5, alpha=1)
# plt.show()

# #### Plot Bean Type
# plot_df = main_data['Bean Type'].fillna(value='Unavailable', method=None, axis=None, inplace=False, limit=None, downcast=None) 
# plot_df = plot_df.value_counts(normalize=True, sort=True, ascending=False, bins=None, dropna=False)*100
# plot_df_data = list(plot_df)
# print(plot_df_data)
# plot_df_uniques = list(plot_df.keys())
# print(plot_df_uniques)
# plt.figure(dpi=dpi_set,figsize=(7.875,3.375))
# plt.xticks(rotation=90)
# plt.xlabel('Bean Type')
# plt.ylabel('Percent [%]')
# plt.scatter(plot_df_uniques,plot_df_data, s=5)
# plt.grid(b=True, which='major', axis='both', linestyle=':', linewidth=0.5, alpha=1)
# plt.show()

# #### Plot Company Location
# plot_df = main_data['Company Location'].fillna(value='Unavailable', method=None, axis=None, inplace=False, limit=None, downcast=None) 
# plot_df = plot_df.value_counts(normalize=True, sort=True, ascending=False, bins=None, dropna=False)*100
# plot_df_data = list(plot_df)
# plot_df_uniques = list(plot_df.keys())
# plt.figure(dpi=dpi_set,figsize=(7.875,3.375))
# plt.xticks(rotation=90)
# plt.xlabel('Company Location')
# plt.ylabel('Percent [%]')
# plt.scatter(plot_df_uniques,plot_df_data, s=6)
# plt.grid(b=True, which='major', axis='both', linestyle=':', linewidth=0.5, alpha=1)
# plt.show()

# #### Plot Company
# plot_df = main_data['Company'].fillna(value='Unavailable', method=None, axis=None, inplace=False, limit=None, downcast=None) 
# plot_df = plot_df.value_counts(normalize=True, sort=True, ascending=False, bins=None, dropna=False)*100
# plot_df_data = list(plot_df)
# plot_df_uniques = list(plot_df.keys())
# plt.figure(dpi=dpi_set,figsize=(7.875,3.375))
# plt.xticks(rotation=90)
# plt.xlabel('Company')
# plt.ylabel('Percent [%]')
# plt.scatter(plot_df_uniques,plot_df_data, s=5)
# plt.grid(b=True, which='major', axis='both', linestyle=':', linewidth=0.5, alpha=1)
# plt.show()

# #### Plot Specific Bean Origin or Bar Name
# plot_df = main_data['Specific Bean Origin or Bar Name'].fillna(value='Unavailable', method=None, axis=None, inplace=False, limit=None, downcast=None) 
# plot_df = plot_df.value_counts(normalize=True, sort=True, ascending=False, bins=None, dropna=False)*100
# plot_df_data = list(plot_df)
# plot_df_uniques = list(plot_df.keys())
# plt.figure(dpi=dpi_set,figsize=(7.875,3.375))
# plt.xticks(rotation=90)
# plt.xlabel('Specific Bean Origin or Bar Name')
# plt.ylabel('Percent [%]')
# plt.scatter(plot_df_uniques,plot_df_data, s=7)
# plt.grid(b=True, which='major', axis='both', linestyle=':', linewidth=0.5, alpha=1)
# plt.show()

# #### Plot Broad Bean Origin
# plot_df = main_data['Broad Bean Origin'].fillna(value='Unavailable', method=None, axis=None, inplace=False, limit=None, downcast=None) 
# plot_df = plot_df.value_counts(normalize=True, sort=True, ascending=False, bins=None, dropna=False)*100
# plot_df_data = list(plot_df)
# plot_df_uniques = list(plot_df.keys())
# plt.figure(dpi=dpi_set,figsize=(11.875,3.375))
# plt.xticks(rotation=90)
# plt.xlabel('Broad Bean Origin')
# plt.ylabel('Percent [%]')
# plt.scatter(plot_df_uniques,plot_df_data, s=10)
# plt.grid(b=True, which='major', axis='both', linestyle=':', linewidth=0.5, alpha=1)
# plt.show()



idx = 0
idy = 0
# fig, axs = plt.subplots(3,3, dpi=120, facecolor='w', edgecolor='k') #figsize=()
# fig.subplots_adjust(hspace = .5, wspace=.001)
# axs = axs.ravel()

# for onefeature in features_list:
#     plot_df = main_data[onefeature].value_counts(normalize=True, sort=True, ascending=False, bins=None, dropna=True)*100
#     plot_df_data = list(plot_df)
#     plot_df_uniques = list(plot_df.keys())
#     plt.figure(dpi=1200)#,figsize=(4,3))
#     # axs[idx].set_xlabel(onefeature)
#     # axs[idx].set_ylabel('Percent [%]')
#     # axs[idx].bar(plot_df_uniques,height=plot_df_data, width=0.17,linewidth=0,edgecolor='w')
#     # axs[idx].set_title(onefeature)
#     # plt.figure(dpi=120,figsize=(4,3))
#     plt.xlabel(onefeature)
#     plt.ylabel('Percent [%]')
#     plt.bar(plot_df_uniques,height=plot_df_data, width=0.17,linewidth=0,edgecolor='w')
#     plt.grid(b=True, which='major', axis='both', linestyle=':', linewidth=0.75, alpha=1)
#     plt.show()
#     idx = idx + 1

# plot_df = main_data.groupby(by='Rating', as_index=False).agg({'ID': pd.Series.nunique})

# ['']

print("\nDone!!!\n")