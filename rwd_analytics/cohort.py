import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from rwd_analytics.lookups import Descendants, Concept


class Cohort():
    def __init__(self, omop_tables, cohort):
        self.cohort = cohort
        self.subjects = cohort.person_id.tolist()
        self.omop_tables = omop_tables

    def demographics(self, show=False):
        df = self.omop_tables['person']
        df = df.loc[df.index.isin(self.subjects)].compute().reset_index()
        df = self.cohort.merge(df, how='left', on='person_id')

        # Getting age and gender information
        df = df[~df['year_of_birth'].isna()]
        df['age_at_index'] = df['index_date'].dt.year - df['year_of_birth']
        gender_map = {8532: 'Female', 8507: 'Male'}
        df['gender_concept_id'] = df['gender_concept_id'].map(gender_map)

        # Getting observation time
        obs_period = self.omop_tables['observation_period']
        obs_period = obs_period.loc[obs_period.index.isin(self.subjects)].compute()
        df = df.merge(obs_period, how='left', on='person_id')
        df['time_before_index'] = (df['index_date'] - df['observation_period_start_date']).dt.days
        df['time_after_index'] = (df['observation_period_end_date'] - df['index_date']).dt.days
        df = df[df['time_before_index'] >= 0]
        df = df[df['time_after_index'] >= 0]
        df['time_before_index'] = df['time_before_index'].astype(int)
        df['time_after_index'] = df['time_after_index'].astype(int)
        df = df[['person_id', 'index_date', 'age_at_index', 'gender_concept_id',
                 'time_before_index', 'time_after_index']]

        if show:
            demo = {}
            demo['age_at_index'] = df.age_at_index.value_counts().to_frame('count') \
                .reset_index().rename(columns={'index': 'age_at_index'})
            demo['gender_concept_id'] = df.gender_concept_id.value_counts().to_frame('count') \
                .reset_index().rename(columns={'index': 'gender_concept_id'})
            demo['index_date'] = df.index_date.dt.year.value_counts().to_frame('count') \
                .reset_index().rename(columns={'index': 'index_date'})
            f, axes = plt.subplots(2, 3, figsize=(13, 5))
            sns.set(style="whitegrid")
            sns.barplot(x="index_date", y="count", data=demo['index_date'], ax=axes[0, 0])
            sns.lineplot(x="age_at_index", y="count", data=demo['age_at_index'], ax=axes[0, 1])
            sns.barplot(x="gender_concept_id", y="count", data=demo['gender_concept_id'],
                        ax=axes[0, 2])
            sns.boxplot(x=df["time_before_index"], ax=axes[1, 0])
            sns.boxplot(x=df["time_after_index"], ax=axes[1, 1])
            plt.setp(axes, yticks=[])
            plt.tight_layout()

        return df


class CohortBuilder():
    def __init__(self, cohort_definition, omop_tables, cohort=None):
        self.omop_tables = omop_tables
        self.person = omop_tables['person']
        self.obs_period = omop_tables['observation_period']
        self.descendants = Descendants()
        self.cohort_definition = cohort_definition

        if cohort is None:
            self.cohort = pd.DataFrame(columns=['person_id', 'index_date'])
        else:
            self.cohort = cohort

    def __call__(self):
        demographic_criteria = self.cohort_definition.setdefault('demographic_criteria', None)
        if demographic_criteria:
            if len(self.cohort) == 0:
                tmp = self.person
            else:
                subjects = self.cohort.person_id.unique().tolist()
                try:
                    tmp = self.person.loc[subjects].compute()
                except:
                    print('Some patients do not have demographic information and will be removed')
                    tmp = self.person.loc[self.person.index.isin(subjects)].compute()

            if demographic_criteria.setdefault('min_year_of_birth', None):
                tmp = tmp[tmp['year_of_birth'] >= demographic_criteria['min_year_of_birth']]
            if demographic_criteria.setdefault('max_year_of_birth', None):
                tmp = tmp[tmp['year_of_birth'] <= demographic_criteria['max_year_of_birth']]
            if demographic_criteria.setdefault('gender_concept_id', None):
                tmp = tmp[tmp['gender_concept_id'].isin(demographic_criteria['gender_concept_id'])]

            try:
                self.cohort = tmp.compute().reset_index()
            except:
                self.cohort = tmp.reset_index()
            self.cohort['index_date'] = pd.to_datetime('1970-01-01')

        for criteria in self.cohort_definition['criteria_list']:
            criteria.setdefault('is_excluded', 0)
            criteria.setdefault('get_descendants', 0)
            criteria.setdefault('mapped', None)
            criteria.setdefault('occurrence_start_date', None)
            criteria.setdefault('occurrence_end_date', None)
            criteria.setdefault('min_occurrence', None)
            criteria.setdefault('min_duration', None)
            criteria.setdefault('is_before_previous_criteria', None)
            criteria.setdefault('is_after_previous_criteria', None)

            tmp = self.omop_tables[criteria['concept_type']]
            tmp = tmp.rename(columns={
                'condition_concept_id': 'concept_id',
                'condition_source_concept_id': 'source_concept_id',
                'condition_start_datetime': 'start_date',
                'drug_concept_id': 'concept_id',
                'drug_source_concept_id': 'source_concept_id',
                'drug_exposure_start_datetime': 'start_date'
            })
            len_cohort = len(self.cohort)
            if len_cohort != 0:
                subjects = self.cohort.person_id.unique().tolist()
                try:
                    tmp = tmp.loc[subjects].compute()
                except:
                    tmp = tmp.loc[tmp.index.isin(subjects)].compute()

            # Filtering by concept ID
            if criteria['concept_id']:
                concept_ids = criteria['concept_id']
                if criteria['mapped'] == 1:
                    print('Criteria: getting source concept IDs '+str(criteria['concept_id']))
                    tmp = tmp[tmp['source_concept_id'].isin(concept_ids)]
                else:
                    if criteria['get_descendants'] == 1:
                        print('Criteria: getting descenant of standard concept IDs '
                              + str(criteria['concept_id']))
                        concept_ids = self.descendants(concept_ids)
                    tmp = tmp[tmp['concept_id'].isin(concept_ids)]

            # Filtering by date
            if criteria['occurrence_start_date']:
                print('Criteria: start after '+str(criteria['occurrence_start_date']))
                tmp = tmp[tmp['start_date'] >= pd.to_datetime(criteria['occurrence_start_date'])]

            if criteria['occurrence_end_date']:
                print('Criteria: start before '+str(criteria['occurrence_end_date']))
                tmp = tmp[tmp['start_date'] <= pd.to_datetime(criteria['occurrence_end_date'])]

            try:
                tmp = tmp.compute()
            except:
                ''

            # Filtering by number of occurrences
            if criteria['min_occurrence']:
                print('Criteria: minimum occurrence '+str(criteria['min_occurrence']))
                tmp = tmp.groupby('person_id').start_date.agg(['min', 'count'])
                tmp = tmp[tmp['count'] >= criteria['min_occurrence']]
                tmp.columns = ['start_date', 'count']

            # Filtering by duration
            if criteria['min_duration']:
                print('Criteria: minimum duration '+str(criteria['min_duration'])+' days')
                tmp = tmp.groupby('person_id').start_date.agg(['min', 'max'])
                tmp['duration'] = (tmp['max'] - tmp['min']).dt.days
                tmp = tmp[tmp['duration'] > criteria['min_duration']]
                tmp = tmp.rename(columns={'min': 'start_date'})

            if (criteria['min_occurrence'] is None) & (criteria['min_duration'] is None):
                tmp = tmp.groupby('person_id').start_date.min()

            tmp = tmp.reset_index()

            if criteria['is_before_previous_criteria']:
                tmp = tmp.merge(self.cohort, on='person_id', how='inner')
                tmp = tmp[tmp['start_date'] < tmp['index_date']]
                del tmp['index_date']

            if criteria['is_after_previous_criteria']:
                tmp = tmp.merge(self.cohort, on='person_id', how='inner')
                tmp = tmp[tmp['start_date'] >= tmp['index_date']]
                del tmp['index_date']

            cohort_temp = tmp.rename(columns={'start_date': 'index_date'}).copy()
            cohort_temp = cohort_temp[['person_id', 'index_date']]

            if len_cohort != 0:
                if criteria['is_excluded'] == 0:
                    self.cohort = pd.concat([self.cohort, cohort_temp])
                    self.cohort = self.cohort.groupby('person_id').agg({
                        'person_id': 'count', 'index_date': 'max'
                        })
                    self.cohort.columns = ['count', 'index_date']
                    self.cohort = self.cohort[self.cohort['count'] >= 2]
                    self.cohort = self.cohort.reset_index()
                    del self.cohort['count']
                else:
                    self.cohort = self.cohort[
                        ~self.cohort['person_id'].isin(cohort_temp.person_id.unique().tolist())]
            else:
                self.cohort = cohort_temp

            len_cohort = len(self.cohort)
            if len_cohort == 0:
                print('There is not enough patients in this cohort, change the criteria.')
                break
            else:
                print('Number of patients: '+str(len_cohort))
                print('*****************')

        obs_period_criteria = self.cohort_definition.setdefault('observation_period_criteria',
                                                                None)
        if obs_period_criteria:
            subject = self.cohort.person_id.unique().tolist()
            try:
                self.obs_period = self.obs_period.loc[subject].compute().reset_index()
            except:
                self.obs_period = self.obs_period.loc[
                    self.obs_period.index.isin(subject)].compute().reset_index()

            self.cohort = self.cohort.merge(self.obs_period, how='left', on='person_id')
            self.cohort['obs_before_index'] = \
                (self.cohort['index_date'] - self.cohort['observation_period_start_date']).dt.days
            self.cohort['obs_after_index'] = \
                (self.cohort['observation_period_end_date'] - self.cohort['index_date']).dt.days

            if obs_period_criteria.setdefault('before_index', None):
                self.cohort = self.cohort[self.cohort['obs_before_index']
                                          >= obs_period_criteria['before_index']]

            if obs_period_criteria.setdefault('after_index', None):
                self.cohort = self.cohort[self.cohort['obs_after_index']
                                          >= obs_period_criteria['after_index']]

        return self.cohort[['person_id', 'index_date']]


def get_distribution(omop_tables, concept_ids, start_date=None, end_date=None, cohort=None):
    """
    Returns nb of occurrences of given concept IDs in the database
    Parameters:
        - omop_tables
        - concepts_ids: list of homogeneous standard or non standard concept ids
        - start_date: format %Y-%m-%d
        - end_date: format %Y-%m-%d
        - cohort: optinal - can accelerate compute
    """
    concept = Concept()
    concept_info = concept(concept_ids)
    domain = concept_info.at[0, 'domain_id']
    standard = concept_info.at[0, 'standard_concept']
    map_domain_to_table = {
        'Drug': 'drug_exposure',
        'Condition': 'condition_occurrence',
        'Procedure': 'procedure_occurrence'
    }
    df = omop_tables[map_domain_to_table[domain]]
    df = df.rename(columns={
        'condition_concept_id': 'concept_id',
        'condition_source_concept_id': 'source_concept_id',
        'condition_start_datetime': 'start_date',
        'drug_concept_id': 'concept_id',
        'drug_source_concept_id': 'source_concept_id',
        'drug_exposure_start_datetime': 'start_date',
        'procedure_concept_id': 'concept_id',
        'procedure_source_concept_id': 'source_concept_id',
        'procedure_datetime': 'start_date'
    })
    if cohort is not None:
        subjects = cohort.person_id.tolist()
        df = df.loc[df.index.isin(subjects)].compute()

    if standard == 'S':
        concept_id_level = 'concept_id'
    else:
        concept_id_level = 'source_concept_id'

    df = df[df[concept_id_level].isin(concept_ids)]

    if start_date:
        df = df[df['start_date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['start_date'] <= pd.to_datetime(end_date)]

    try:
        df = df.compute().reset_index()
    except:
        df = df.reset_index()

    df = df.groupby(concept_id_level).agg({'person_id': ['count', pd.Series.nunique]})
    df.columns = df.columns.droplevel()
    df = df.reset_index()
    df = df.rename(columns={
        concept_id_level: 'concept_id',
        'nunique': 'n_unique_patients',
        'count': 'n_records'
    })
    df = concept.get_info(df, ['concept_code', 'concept_name', 'vocabulary_id'])
    return df[['concept_id', 'concept_code', 'vocabulary_id',
               'concept_name', 'n_unique_patients', 'n_records']]
