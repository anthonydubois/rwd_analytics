import pandas as pd
import numpy as np
import torch
import random

# sudo docker-compose exec --user root  notebook bash
# pip install -U scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def check_for_na(df):
    print(len(df))
    df.isna().sum()
    df.dropna()
    print(len(df))
    return df


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_matching_pairs(treated_df_input, non_treated_df_input, distance_max=2.5, scaler=True):
    treated_df = treated_df_input.copy()
    non_treated_df = non_treated_df_input.copy()
    del treated_df['person_id']
    del non_treated_df['person_id']
    treated_x = treated_df.values
    non_treated_x = non_treated_df.values
    if scaler:
        scaler = StandardScaler()
    if scaler:
        scaler.fit(treated_x)
        treated_x = scaler.transform(treated_x)
        non_treated_x = scaler.transform(non_treated_x)

    nbrs = NearestNeighbors(n_neighbors=len(treated_df), algorithm='ball_tree').fit(treated_x)
    distances, indices = nbrs.kneighbors(non_treated_x)
    t = pd.DataFrame(distances, index=['distances'])
    t = t.append(pd.DataFrame(indices, index=['indices']))
    t = t.T
    t['indices'] = t['indices'].astype(int)
    near_patients = t[t['distances'] < distance_max].indices.unique().tolist()
    matched = treated_df_input.loc[near_patients]
    return matched.reset_index(drop=True)
