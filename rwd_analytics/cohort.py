import pandas as pd
import dask.dataframe as dd
import numpy as np
import math

from rwd_analytics.lookups import Descendants


class CohortBuilder():
    """
    cohort_criteria is a json that can have different arguments:
    - 
    Warning: Put inclusion criteria before exclusion criteria
    """
    def __init__(self, cohort_criteria, drug_exposure, condition_occurrence, person, observation_period, cohort=None):
        self.conditions = condition_occurrence[
            ['condition_concept_id', 'condition_start_datetime']]
        self.conditions = self.conditions.rename(columns={
            'condition_concept_id':'concept_id',
            'condition_start_datetime':'cohort_start_date'
        })
        self.drugs = drug_exposure[
            ['drug_concept_id', 'drug_exposure_start_datetime']]
        self.drugs = self.drugs.rename(columns={
            'drug_concept_id':'concept_id',
            'drug_exposure_start_datetime':'cohort_start_date'
        })
        self.person = person[['gender_concept_id', 'year_of_birth']]
        self.obs_period = observation_period[
            ['observation_period_start_date', 'observation_period_end_date']]
        self.descendants = Descendants()
        self.cohort_criteria = cohort_criteria
        if cohort is None:
            self.cohort = pd.DataFrame(columns=['person_id', 'cohort_start_date'])
        else:
            self.cohort = cohort

    def __subjects_with_criteria(self, tmp, criteria):
        concept_ids = criteria['concept_id']
        descendant = criteria['descendant']
        #mapped = criteria['mapped']
        
        if descendant == 1:
            concept_ids = self.descendants(concept_ids)

        tmp = tmp[tmp['concept_id'].isin(concept_ids)]
        return tmp

    def __attributes_selection(self, tmp, criteria):
        attributes = criteria['attributes']
        if len(attributes) != 0:
            for attribute in attributes:
                if attribute['type'] == 'occurrence':
                    tmp = tmp.groupby('person_id').cohort_start_date.agg(['min', 'count'])
                    tmp = tmp[tmp['count'] >= attribute['min']]
                    tmp.columns = ['cohort_start_date', 'count']
        else:
            tmp = tmp.groupby('person_id').cohort_start_date.min()
        
        tmp = tmp.reset_index()
        return tmp[['person_id', 'cohort_start_date']]

    def __call__(self):
        for criteria in self.cohort_criteria['criteria']:
            concept_type = criteria['concept_type']
            excluded = criteria['excluded']
            attributes = criteria['attributes']

            if concept_type == 'year_of_birth':
                cohort_temp = self.person[['year_of_birth']]
                for attribute in attributes:
                    if attribute['type'] == 'inferior or equal':
                        cohort_temp = cohort_temp[cohort_temp['year_of_birth'] <= attribute['bound']]
                    if attribute['type'] == 'superior or equal':
                        cohort_temp = cohort_temp[cohort_temp['year_of_birth'] >= attribute['bound']]

            if concept_type == 'gender':
                cohort_temp = self.person[self.person['gender_concept_id'].isin(criteria['concept_id'])]

            if concept_type in ['year_of_birth', 'gender']:
                cohort_temp['cohort_start_date'] = pd.to_datetime('1990-01-01', format='%Y-%m-%d')
            
            if concept_type == 'condition':
                cohort_temp = self.__subjects_with_criteria(self.conditions, criteria)
            
            if concept_type == 'drug':
                cohort_temp = self.__subjects_with_criteria(self.drugs, criteria)

            len_cohort = len(self.cohort)
            if len_cohort != 0:
                subjects = self.cohort.person_id.unique().tolist()
                cohort_temp = cohort_temp.loc[cohort_temp.index.isin(subjects)]
                cohort_temp = cohort_temp.compute()
            else:
                cohort_temp = cohort_temp.compute()

            cohort_temp = cohort_temp.reset_index()
            cohort_temp = cohort_temp[['person_id', 'cohort_start_date']]
            cohort_temp = self.__attributes_selection(cohort_temp, criteria)

            len_cohort = len(self.cohort)
            if len_cohort != 0:
                if excluded == 0:
                    self.cohort = pd.concat([self.cohort, cohort_temp])
                    self.cohort = self.cohort.groupby('person_id').agg({
                        'person_id':'count', 'cohort_start_date':'max'
                        })
                    self.cohort.columns = ['count', 'cohort_start_date']
                    self.cohort = self.cohort[self.cohort['count']>=2]
                    self.cohort = self.cohort.reset_index()
                    del self.cohort['count']
                else:
                    self.cohort = self.cohort[~self.cohort['person_id'].isin(cohort_temp.person_id.unique().tolist())]
            else:
                self.cohort = cohort_temp

            len_cohort = len(self.cohort)
            if len_cohort == 0:
                print('There is not enough patients in this cohort, change the criteria.')
                break
            else:
                print('Criteria: '+concept_type+' '+str(criteria['concept_id']))
                print('Exclusion: '+str(excluded))
                print('Number of patients: '+str(len_cohort))
                print('*****************')

        self.cohort['person_id'] = self.cohort['person_id'].astype(int)
        try:
            self.cohort_criteria['observation_period']
        except KeyError:
            return self.cohort

        subject = self.cohort.person_id.unique().tolist()
        self.obs_period = self.obs_period.loc[subject].compute().reset_index()
        self.cohort = self.cohort.merge(self.obs_period, how='left', on='person_id')
        self.cohort['obs_before_index'] = (self.cohort['cohort_start_date'] \
            - self.cohort['observation_period_start_date']).dt.days
        self.cohort['obs_after_index'] = (self.cohort['observation_period_end_date'] \
            - self.cohort['cohort_start_date']).dt.days
        self.cohort = self.cohort[self.cohort['obs_before_index'] \
            >= self.cohort_criteria['observation_period']['before_index']]
        self.cohort = self.cohort[self.cohort['obs_after_index'] \
            >= self.cohort_criteria['observation_period']['after_index']]
        
        return self.cohort[['person_id', 'cohort_start_date']]
