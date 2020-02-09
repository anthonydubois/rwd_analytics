import pandas as pd
import dask.dataframe as dd
import numpy as np
import math

from rwd_analytics.lookups import Descendants


class CohortBuilder():
    """
    First inclusion then exclusion criteria
    """
    def __init__(self, cohort_criteria, drug_exposure, condition_occurrence, person):
        self.conditions = condition_occurrence[
            ['person_id', 'condition_concept_id', 'condition_start_datetime']]
        self.conditions = self.conditions.rename(columns={
            'condition_concept_id':'concept_id',
            'condition_start_datetime':'cohort_start_date'
        })
        self.drugs = drug_exposure[
            ['person_id', 'drug_concept_id', 'drug_exposure_start_datetime']]
        self.drugs = self.drugs.rename(columns={
            'drug_concept_id':'concept_id',
            'drug_exposure_start_datetime':'cohort_start_date'
        })
        self.person = person[['person_id', 'gender_concept_id', 'year_of_birth']]
        self.descendants = Descendants()
        self.cohort_criteria = cohort_criteria
        self.cohort = pd.DataFrame(columns=['person_id', 'cohort_start_date'])

    def __subjects_with_criteria(self, tmp, criteria):
        concept_ids = criteria['concept_id']
        descendant = criteria['descendant']
        #mapped = criteria['mapped']
        attributes = criteria['attributes']
        
        if descendant == 1:
            concept_ids = self.descendants(concept_ids)

        tmp = tmp[tmp['concept_id'].isin(concept_ids)]
        
        if len(attributes) != 0:
            for attribute in attributes:
                if attribute['type'] == 'occurrence':
                    t = tmp.groupby('person_id').cohort_start_date.nunique().reset_index()
                    t.columns = ['person_id', 'count']
                    t = t[t['count'] >= attribute['feature']]
                    tmp = tmp[tmp['person_id'].isin(t.person_id.tolist())]

        return tmp

    def __call__(self):
        for criteria in self.cohort_criteria['criteria']:
            concept_type = criteria['concept_type']
            excluded = criteria['excluded']

            if concept_type == 'gender':
                cohort_temp = self.person[self.person['gender_concept_id'].isin(criteria['concept_id'])]
                cohort_temp['cohort_start_date'] = pd.to_datetime('1990-01-01', format='%Y-%m-%d')
            
            if concept_type == 'condition':
                tmp = self.conditions
                cohort_temp = self.__subjects_with_criteria(tmp, criteria)
            
            if concept_type == 'drug':
                tmp = self.drugs
                cohort_temp = self.__subjects_with_criteria(tmp, criteria)

            len_cohort = len(self.cohort)
            if len_cohort != 0:
                subjects = self.cohort.person_id.unique().compute().tolist()
                cohort_temp = cohort_temp[cohort_temp['person_id'].isin(subjects)]

            cohort_temp = cohort_temp.compute()

            if excluded == 0:
                cohort_temp = cohort_temp[['person_id', 'cohort_start_date']]
                self.cohort = pd.concat([self.cohort, cohort_temp])
                self.cohort = self.cohort.groupby('person_id').cohort_start_date.max()
                self.cohort = self.cohort.to_frame().reset_index()
            else:
                self.cohort = self.cohort[~self.cohort['person_id'].isin(cohort_temp.person_id.unique().tolist())]
            
            len_cohort = len(self.cohort)
            if len_cohort == 0:
                print('There is not enough patients in this cohort, change the criteria.')
                break
            else:
                print('criteria: '+str(criteria['concept_id'])+'- Number of patients: '+str(len_cohort))
        
        return self.cohort
