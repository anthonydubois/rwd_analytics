import pandas as pd
import dask.dataframe as dd
import numpy as np
import math

from rwd_analytics.lookups import Descendants


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
        demographic_criteria = self.cohort_definition['demographic_criteria']
        notnone = {k:v for k, v in demographic_criteria.items() if v is not None}
        if len(notnone) != 0:
            if len(self.cohort) == 0:
                tmp = self.person
            else:
                tmp = self.person.loc[self.person.index.isin(self.cohort.person_id.unique().tolist())]
            
            if demographic_criteria['min_year_of_birth']:
                tmp = tmp[tmp['year_of_birth'] >= demographic_criteria['min_year_of_birth']]
            if demographic_criteria['max_year_of_birth']:
                tmp = tmp[tmp['year_of_birth'] <= demographic_criteria['max_year_of_birth']]
            if demographic_criteria['gender_concept_id']:
                tmp = tmp[tmp['gender_concept_id'].isin(demographic_criteria['gender_concept_id'])]
                
            self.cohort = tmp.compute().reset_index()
            self.cohort['index_date'] = pd.to_datetime('1970-01-01')

        for criteria in self.cohort_definition['criteria_list']:
            tmp = self.omop_tables[criteria['concept_type']]
            tmp = tmp.rename(columns={
                'condition_concept_id':'concept_id',
                'condition_start_datetime':'start_date',
                'drug_concept_id':'concept_id',
                'drug_exposure_start_datetime':'start_date'
            })
            len_cohort = len(self.cohort)
            if len_cohort != 0:
                subjects = self.cohort.person_id.unique().tolist()
                tmp = tmp.loc[tmp.index.isin(subjects)]
            
            # Filtering by concept ID
            if criteria['concept_id']:
                concept_ids = criteria['concept_id']
                if criteria['get_descendants'] == 1:
                    concept_ids = self.descendants(concept_ids)
                tmp = tmp[tmp['concept_id'].isin(concept_ids)]
            
            # Filtering by date
            if criteria['occurrence_start_date']:
                tmp = tmp[tmp['start_date'] >= pd.to_datetime(criteria['occurrence_start_date'])]
                
            if criteria['occurrence_end_date']:
                tmp = tmp[tmp['start_date'] <= pd.to_datetime(criteria['occurrence_end_date'])]
                
            tmp = tmp.compute()

            # Filtering by number of occurrences
            if criteria['min_occurrence']:
                tmp = tmp.groupby('person_id').start_date.agg(['min', 'count'])
                tmp = tmp[tmp['count'] >= criteria['min_occurrence']]
                tmp.columns = ['start_date', 'count']

            # Filtering by duration
            if criteria['min_duration']:
                tmp = tmp.groupby('person_id').start_date.agg(['min', 'max'])
                tmp['duration'] = (tmp['max'] - tmp['min']).dt.days
                tmp = tmp[tmp['duration'] > criteria['min_duration']]
                tmp = tmp.rename(columns={'min':'start_date'})
            
            if (criteria['min_occurrence'] is None) & (criteria['min_duration'] is None):
                tmp = tmp.groupby('person_id').start_date.min()
                
            tmp = tmp.reset_index()
                
            if criteria['is_before_previous_criteria']:
                tmp = tmp.merge(self.cohort, on ='person_id', how='inner')
                tmp = tmp[tmp['start_date'] < tmp['index_date']]
                del tmp['index_date']
                    
            if criteria['is_after_previous_criteria']:
                tmp = tmp.merge(self.cohort, on ='person_id', how='inner')
                tmp = tmp[tmp['start_date'] >= tmp['index_date']]
                del tmp['index_date']
                
            cohort_temp = tmp.rename(columns={'start_date':'index_date'}).copy()
            cohort_temp = cohort_temp[['person_id', 'index_date']]
                
            if len_cohort != 0:
                if criteria['is_excluded'] == 0:
                    self.cohort = pd.concat([self.cohort, cohort_temp])
                    self.cohort = self.cohort.groupby('person_id').agg({
                        'person_id':'count', 'index_date':'max'
                        })
                    self.cohort.columns = ['count', 'index_date']
                    self.cohort = self.cohort[self.cohort['count']>=2]
                    self.cohort = self.cohort.reset_index()
                    del self.cohort['count']
                else:
                    self.cohort = self.cohort[~self.cohort['person_id'] \
                                              .isin(cohort_temp.person_id.unique().tolist())]
            else:
                self.cohort = cohort_temp

            len_cohort = len(self.cohort)
            if len_cohort == 0:
                print('There is not enough patients in this cohort, change the criteria.')
                break
            else:
                print('Criteria: '+str(criteria['concept_id']))
                print('Number of patients: '+str(len_cohort))
                print('*****************')
        
        obs_period_criteria = self.cohort_definition['observation_period_criteria']        
        notnone = {k:v for k, v in obs_period_criteria.items() if v is not None}
        if len(notnone) != 0:
            subject = self.cohort.person_id.unique().tolist()
            self.obs_period = self.obs_period.loc[subject].compute().reset_index()
            self.cohort = self.cohort.merge(self.obs_period, how='left', on='person_id')
            self.cohort['obs_before_index'] = (self.cohort['index_date'] \
                - self.cohort['observation_period_start_date']).dt.days
            self.cohort['obs_after_index'] = (self.cohort['observation_period_end_date'] \
                - self.cohort['index_date']).dt.days
            
            if obs_period_criteria['before_index']:
                self.cohort = self.cohort[self.cohort['obs_before_index'] \
                    >= obs_period_criteria['before_index']]
            
            if obs_period_criteria['after_index']:
                self.cohort = self.cohort[self.cohort['obs_after_index'] \
                    >= obs_period_criteria['after_index']]
        
        return self.cohort[['person_id', 'index_date']]
    

class CohortCharacterization():
    def __init__(self, characterization_definition, cohort, omop_tables):
        self.tables_for_characterization = ['drug_exposure', 'condition_occurrence', 'procedure_occurrence']
        self.characterization_table = {}
        for table in self.tables_for_characterization:
            tmp = omop_tables[table]
            tmp = tmp.rename(columns={
                'condition_concept_id':'concept_id',
                'condition_start_datetime':'start_date',
                'drug_concept_id':'concept_id',
                'drug_exposure_start_datetime':'start_date',
                'procedure_concept_id':'concept_id',
                'procedure_datetime':'start_date'
            })
            tmp = tmp.loc[tmp.index.isin(cohort.person_id.tolist())]
            tmp = tmp.reset_index()
            tmp['key'] = tmp['person_id'].astype(str)+'_'+tmp['start_date'].astype(str)
            self.characterization_table[table] = tmp.compute()
    
        key_list = self.characterization_table[characterization_definition['concept_type']]
        key_list = key_list[key_list['concept_id'].isin(characterization_definition['concept_id'])]
        self.key_list = key_list['key'].unique().tolist()

    def __call__(self):
        for table in self.tables_for_characterization:
            tmp = self.characterization_table[table][self.characterization_table[table]['key'].isin(self.key_list)]
            t_characterization_df = tmp.groupby('concept_id').agg({'person_id':[pd.Series.nunique, 'count']})
            t_characterization_df.columns = t_characterization_df.columns.droplevel()
            t_characterization_df = t_characterization_df.reset_index()
            characterization_df = characterization_df.append(t_characterization_df)

        return characterization_df