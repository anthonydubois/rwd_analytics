import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from rwd_analytics.lookups import Descendants, ComorbidConditions


class FeaturesSelection():
    def __init__(self, cohort, features, omop_tables):
        self.X = cohort
        self.subjects = self.X.person_id.unique().tolist()
        self.number_of_subjects = len(self.subjects)
        self.features = features
        self.drug_exposure = omop_tables['drug_exposure']
        self.condition_occurrence = omop_tables['condition_occurrence']
        self.condition_occurrence = self.condition_occurrence.rename(columns={
            'condition_concept_id':'concept_id',
            'condition_start_datetime':'start_date'
        })
        self.visit_occurrence = omop_tables['visit_occurrence']
        self.measurement = omop_tables['measurement']
        self.procedure = omop_tables['procedure_occurrence']
        self.comorbidities = ComorbidConditions()
        self.person = omop_tables['person']
        
    def __clean_features_by_occurrences(self):
        min_feat_occurrence = self.features['time_windows']['minimum']*self.number_of_subjects
        for (columnName, columnData) in self.X.iteritems():
            if columnName in ['person_id', 'index_date']:
                continue
                
            if columnData.values.sum() < min_feat_occurrence:
                del self.X[columnName]

    def __non_time_bound_features(self):
        print('Getting non time bound features')
        try:
            self.person = self.person.loc[self.subjects].compute()
        except:
            self.person = self.person.loc[self.person.index.isin(self.subjects)].compute()
        
        self.X = self.X.merge(self.person, how='left', on='person_id')
        print('We had to drop the following patients because they did not have a record in PERSON:')
        print(self.X[self.X['year_of_birth'].isna()].person_id.tolist())
        self.X = self.X[~self.X['year_of_birth'].isna()]
        self.X['age_at_index'] = self.X['index_date'].dt.year - self.X['year_of_birth']
        print(self.X)

        if self.features['non_time_bound']['age_group'] == 1:
            age_groups = ['00-04', '05-09', '10-14', '15-19', '20-24', '25-29',
                          '30-34', '35-39', '40-44', '45-49', '50-54', '55-59',
                          '60-64', '65-69', '70-74', '75-79', '80-84', '85-89',
                          '90-94', '95+']
            i = 5
            for age_group in age_groups:
                s = self.X['age_at_index']
                self.X[age_group] = 1
                self.X[age_group] = self.X[age_group].where((s < i) & (s >= i - 5), 0)
                i = i + 5

        del self.X['year_of_birth']
        self.X['age_at_index'] = self.X['age_at_index'].astype(int)

        if self.features['non_time_bound']['age_at_index'] == 0:
            del self.X['age_at_index']

        if self.features['non_time_bound']['gender'] == 1:
            self.X['gender = female'] = 1
            self.X['gender = female'] = self.X['gender = female'].where(self.X['gender_concept_id'] == 8532, 0)

        del self.X['gender_concept_id']

    def __feature_generator_timely(self, df, feature_name, time_window):
        tmp = df[df['time_to_index'] <= self.features['time_windows'][time_window]]

        if feature_name == 'visit_count_':
            tmp = tmp.groupby('person_id').start_date.count().to_frame()
            tmp.columns = [feature_name+time_window]
            self.X = self.X.merge(tmp, on='person_id', how='left')
            self.X[feature_name+time_window] = self.X[feature_name+time_window].fillna(0).astype(int)
            
        elif feature_name == 'measurement_range_group':
            tmp = tmp[['person_id', 'concept_id']]
            concepts = tmp.concept_id.unique().tolist()
            for concept in concepts:
                subjects = tmp[tmp['concept_id']==concept].person_id.unique().tolist()
                if len(subjects) > self.features['time_windows']['minimum']*self.number_of_subjects:
                    t = str(concept)+'_'+time_window
                    self.X[t] = 1
                    self.X[t] = self.X[t].where(self.X['person_id'].isin(subjects), 0)
                    
            tmp = tmp.groupby('person_id').start_date.count().to_frame()
            tmp.columns = [feature_name+time_window]
            self.X = self.X.merge(tmp, on='person_id', how='left')
            self.X[feature_name+time_window] = self.X[feature_name+time_window].fillna(0).astype(int)
            
        elif feature_name == 'comorbid_condition':
            for row in self.comorbidities().itertuples(index=True, name='Pandas'):
                f_name = getattr(row, "COMMORBIDITIES").lower().replace(' ', '_')+'_'+time_window
                comorbid_concepts = getattr(row, "CONCEPT_ID").replace('[', '').replace(']', '').split(', ')
                comorbid_tmp = tmp[tmp['concept_id'].isin(comorbid_concepts)]
                subjects = comorbid_tmp.person_id.unique().tolist()
                self.X[f_name] = 1
                self.X[f_name] = self.X[f_name].where(self.X['person_id'].isin(subjects), 0)
            
        else:
            tmp = tmp[['person_id', 'concept_id']]
            concepts = tmp.concept_id.unique().tolist()
            for concept in concepts:
                subjects = tmp[tmp['concept_id']==concept].person_id.unique().tolist()
                if len(subjects) > self.features['time_windows']['minimum']*self.number_of_subjects:
                    t = str(concept)+'_'+time_window
                    self.X[t] = 1
                    self.X[t] = self.X[t].where(self.X['person_id'].isin(subjects), 0)

    def __feature_generator(self, df, feature_name, time_features):
        try:
            # Much faster
            df = df.loc[self.subjects].compute()
        except:
            # In case of all patients not in index
            df = df.loc[df.index.isin(self.subjects)].compute()

        df = df.merge(self.X, how='inner', on='person_id')
        df['time_to_index'] = (df['index_date'] - df['start_date']).dt.days
        df = df[df['time_to_index'] > 0]

        if time_features[0] == 1:
            self.__feature_generator_timely(df, feature_name, 'inf')

        if time_features[1] == 1:
            self.__feature_generator_timely(df, feature_name, 'long')

        if time_features[2] == 1:
            self.__feature_generator_timely(df, feature_name, 'med')

        if time_features[3] == 1:
            self.__feature_generator_timely(df, feature_name, 'short')
    
    def __call__(self):
        if self.features['non_time_bound']['age_group'] + \
           self.features['non_time_bound']['age_at_index'] + \
           self.features['non_time_bound']['gender'] != 0:
           self.__non_time_bound_features()

        time_features = self.features['time_bound']['visit_count']
        if sum(time_features) > 0:
            print('Getting visit count features')
            feature_name = 'visit_count_'
            df = self.visit_occurrence[['visit_start_datetime']]
            df = df.rename(columns={
                'visit_start_datetime':'start_date'
            })
            self.__feature_generator(df, feature_name, time_features)
        
        time_features = self.features['time_bound']['condition']
        if sum(time_features) > 0:
            print('Getting condition features')
            feature_name = 'condition'
            df = self.condition_occurrence
            df = df.rename(columns={
                'condition_concept_id':'concept_id',
                'condition_start_datetime':'start_date'
            })
            self.__feature_generator(df, feature_name, time_features)
            
        time_features = self.features['time_bound']['comorbid_condition']
        if sum(time_features) > 0:
            print('Getting comorbid_condition features')
            feature_name = 'comorbid_condition'
            df = self.condition_occurrence
            self.__feature_generator(df, feature_name, time_features)
            
        time_features = self.features['time_bound']['procedure']
        if sum(time_features) > 0:
            print('Getting procedure features')
            feature_name = 'procedure'
            df = self.procedure[['procedure_concept_id', 'procedure_datetime']]
            df = df.rename(columns={
                'procedure_concept_id':'concept_id',
                'procedure_datetime':'start_date'
            })
            self.__feature_generator(df, feature_name, time_features)
            
        time_features = self.features['time_bound']['measurement']
        if sum(time_features) > 0:
            print('Getting measurement features')
            feature_name = 'measurement'
            df = self.measurement[['measurement_concept_id', 'measurement_datetime']]
            df = df.rename(columns={
                'measurement_concept_id':'concept_id',
                'measurement_datetime':'start_date'
            })
            self.__feature_generator(df, feature_name, time_features)
            
        time_features = self.features['time_bound']['measurement_range_group']
        if sum(time_features) > 0:
            print('Getting measurement features')
            feature_name = 'measurement_range_group'
            df = self.measurement[
                ['person_id', 'measurement_concept_id', 'measurement_datetime',
                 'value_as_number', 'range_low', 'range_high']
            ]
            df = df[~df['value_as_number'].isna()]
            df = df.rename(columns={
                'measurement_concept_id':'concept_id',
                'measurement_datetime':'start_date'
            })
            self.__feature_generator(df, feature_name, time_features)
            
        time_features = self.features['time_bound']['measurement_value']
        if sum(time_features) > 0:
            print('Getting measurement features')
            feature_name = 'measurement_value'
            df = self.measurement[
                ['person_id', 'measurement_concept_id', 'measurement_datetime', 'value_as_number']]
            df = df[~df['value_as_number'].isna()]
            df = df.rename(columns={
                'measurement_concept_id':'concept_id',
                'measurement_datetime':'start_date'
            })
            self.__feature_generator(df, feature_name, time_features)

        time_features = self.features['time_bound']['drug']
        if sum(time_features) > 0:
            print('Getting drug features')
            feature_name = 'drug'
            df = self.drug_exposure[['drug_concept_id', 'drug_exposure_start_datetime']]
            df = df.rename(columns={
                'drug_concept_id':'concept_id',
                'drug_exposure_start_datetime':'start_date'
            })
            self.__feature_generator(df, feature_name, time_features)
            
        self.X = self.X.dropna()
        self.__clean_features_by_occurrences()
            
        print('Number of features: '+str(len(self.X.columns)-2))

        return self.X


def time_at_risk(X, cohort_at_risk, cohort_target, time_at_risk = 0):
    cohort_at_risk = cohort_at_risk.merge(cohort_target, how='inner', on='person_id')
    if time_at_risk != 0:
        cohort_at_risk['time_to_event'] = cohort_at_risk['index_date_y'] \
                                          - cohort_at_risk['index_date_x']
        cohort_at_risk = cohort_at_risk[cohort_at_risk['time_to_event'].dt.days < time_at_risk]
    cohort_at_risk = cohort_at_risk.person_id.unique().tolist()
    print('Subject at risk:'+ str(len(cohort_at_risk)))
    X['target'] = 1
    X['target'] = X['target'].where(X['person_id'].isin(cohort_at_risk), 0)
    print('Number of patients in final cohort: '+str(len(X)))
    del X['person_id']
    del X['index_date']
    return X


def get_features_scores(df, n_features):
    X = df.iloc[:,0:n_features]  #independent columns
    y = df.iloc[:,-1]    #target column i.e price range
    
    #apply SelectKBest class to extract top 10 best features
    bestfeatures = SelectKBest(score_func=chi2, k=n_features)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)

    # naming the dataframe columns and rounding results
    featureScores.columns = ['Specs', 'Score']
    featureScores['Score'] = featureScores['Score'].round(2)
    return featureScores.nlargest(n_features, 'Score')


class CovariateSelection():
    def __init__(self, algorithm, X, y, num_feats):
        """
        algorithm:
        - Pearson Correlation: 'pearson_correlation'
        - Chi-Squared: 'chi_squared'
        - Recursive Feature Elimination: 'recursive_feature_elimination'
        - Lasso: SelectFromModel: 'lasso'
        - Random Forest: SelectFromModel: 'random_forest'
        - LightGBM: SelectFromModel:: 'lgbc'
        - All: 'all'
        """
        self.X = X
        self.X_norm = MinMaxScaler().fit_transform(self.X)
        self.y = y
        self.num_feats = num_feats
        self.algorithm = algorithm
        self.feature_name = self.X.columns.tolist()

    def __cor_selector(self):
        cor_list = []
        # calculate the correlation with y for each feature
        for i in self.X.columns.tolist():
            cor = np.corrcoef(self.X[i], self.y)[0, 1]
            cor_list.append(cor)
        # replace NaN with 0
        cor_list = [0 if np.isnan(i) else i for i in cor_list]
        # feature name
        cor_feature = self.X.iloc[:,np.argsort(np.abs(cor_list))[-self.num_feats:]].columns.tolist()
        # feature selection? 0 for not select, 1 for select
        cor_support = [True if i in cor_feature else False for i in self.feature_name]
        return cor_support, cor_feature
    
    
    def __call__(self):
        if self.algorithm in ['pearson_correlation', 'all']:
            cor_support, cor_feature = self.__cor_selector()
            print(str(len(cor_feature)), 'selected features')

        if self.algorithm in ['chi_squared', 'all']:
            chi_selector = SelectKBest(chi2, k=self.num_feats)
            chi_selector.fit(self.X_norm, self.y)
            chi_support = chi_selector.get_support()
            chi_feature = self.X.loc[:,chi_support].columns.tolist()
            print(str(len(chi_feature)), 'selected features')

        if self.algorithm in ['recursive_feature_elimination', 'all']:
            rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=self.num_feats, step=10, verbose=5)
            rfe_selector.fit(self.X_norm, self.y)
            rfe_support = rfe_selector.get_support()
            rfe_feature = self.X.loc[:,rfe_support].columns.tolist()
            print(str(len(rfe_feature)), 'selected features')

        if self.algorithm in ['lasso', 'all']:
            embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l1"), max_features=self.num_feats)
            embeded_lr_selector.fit(self.X_norm, self.y)
            embeded_lr_support = embeded_lr_selector.get_support()
            embeded_lr_feature = self.X.loc[:,embeded_lr_support].columns.tolist()
            print(str(len(embeded_lr_feature)), 'selected features')

        if self.algorithm in ['random_forest', 'all']:
            embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=self.num_feats)
            embeded_rf_selector.fit(self.X, self.y)

            embeded_rf_support = embeded_rf_selector.get_support()
            embeded_rf_feature = self.X.loc[:,embeded_rf_support].columns.tolist()
            print(str(len(embeded_rf_feature)), 'selected features')

        if self.algorithm in ['lgbc', 'all']:
            lgbc=LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
            reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)

            embeded_lgb_selector = SelectFromModel(lgbc, max_features=self.num_feats)
            embeded_lgb_selector.fit(self.X, self.y)

            embeded_lgb_support = embeded_lgb_selector.get_support()
            embeded_lgb_feature = self.X.loc[:,embeded_lgb_support].columns.tolist()
            print(str(len(embeded_lgb_feature)), 'selected features')

        if self.algorithm in ['all']:
            # put all selection together
            feature_selection_df = pd.DataFrame({'Feature':self.feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support,
                                                 'Logistics':embeded_lr_support, 'Random Forest':embeded_rf_support, 'LightGBM':embeded_lgb_support})
            # count the selected times for each feature
            feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
            # display the top 100
            feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
            feature_selection_df.index = range(1, len(feature_selection_df)+1)
            return feature_selection_df.head(self.num_feats)