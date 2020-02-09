import pandas as pd
import dask.dataframe as dd
import pytest

from rwd_analytics.cohort import CohortBuilder
from rwd_analytics.features_selection import FeaturesSelection


cohort = pd.DataFrame({
    'cohort_definition_id':[1, 1, 1, 1, 1, 2, 2, 2],
    'subject_id':[1, 2, 3, 4, 5, 1, 2, 3],
    'cohort_start_date':[
        pd.to_datetime('2018-01-01'),
        pd.to_datetime('2018-01-01'),
        pd.to_datetime('2018-01-01'),
        pd.to_datetime('2018-01-01'),
        pd.to_datetime('2018-01-01'),
        pd.to_datetime('2019-01-01'),
        pd.to_datetime('2018-05-01'),
        pd.to_datetime('2018-03-01'),
    ]
})

person = pd.DataFrame({
    'person_id':[1, 2, 3, 4, 5],
    'gender_concept_id':[8532, 8507, 8532, 8507, 8507],
    'year_of_birth':[1990, 2000, 2010, 1970, 1960]
})
condition_occurrence = pd.DataFrame({
    'person_id':[1, 1, 1, 1, 2, 2],
    'condition_concept_id':[44831230, 2, 3, 4, 44831230, 2],
    'condition_start_datetime':[
        pd.to_datetime('2017-12-10'),
        pd.to_datetime('2017-12-10'),
        pd.to_datetime('2017-12-10'),
        pd.to_datetime('2017-12-10'),
        pd.to_datetime('2017-12-10'),
        pd.to_datetime('2017-12-10'),
    ]
})
drug_exposure = pd.DataFrame({
    'person_id':[1, 1, 1, 1, 2, 2],
    'drug_concept_id':[10, 20, 30, 40, 10, 20],
    'drug_exposure_start_datetime':[
        pd.to_datetime('2017-12-10'),
        pd.to_datetime('2017-12-10'),
        pd.to_datetime('2017-12-10'),
        pd.to_datetime('2017-12-10'),
        pd.to_datetime('2017-12-10'),
        pd.to_datetime('2017-12-10'),
    ]
})

visit_occurrence = pd.DataFrame({
    'person_id':[1],
    'visit_start_datetime':[
        pd.to_datetime('2017-12-10')
    ]
})
visit_occurrence = dd.from_pandas(visit_occurrence, npartitions=1)
person = dd.from_pandas(person, npartitions=1)
condition_occurrence = dd.from_pandas(condition_occurrence, npartitions=1)
drug_exposure = dd.from_pandas(drug_exposure, npartitions=1)
measurement = pd.DataFrame()
procedure = pd.DataFrame()
measurement = dd.from_pandas(measurement, npartitions=1)
procedure = dd.from_pandas(procedure, npartitions=1)


class TestCohort():
    def test_cohort_builder_gender(self):
        cohort_criteria = {
            'criteria':[
                {
                    'concept_type':'gender',
                    'concept_id':[8507],
                    'excluded':0,
                    'descendant':0,
                    'mapped':0,
                    'attributes':[]
                }
            ]
        }
        output = CohortBuilder(cohort_criteria, drug_exposure, condition_occurrence, person)()
        expected = pd.DataFrame({
            'person_id':[2, 4, 5],
            'cohort_start_date':[
                pd.to_datetime('1990-01-01'),
                pd.to_datetime('1990-01-01'),
                pd.to_datetime('1990-01-01')
            ]
        })
        pd.testing.assert_frame_equal(output, expected)

    def test_cohort_builder_condition(self):
        cohort_criteria = {
            'criteria':[
                {
                    'concept_type':'condition',
                    'concept_id':[3],
                    'excluded':0,
                    'descendant':0,
                    'mapped':0,
                    'attributes':[]
                }
            ]
        }
        output = CohortBuilder(cohort_criteria, drug_exposure, condition_occurrence, person)()
        expected = pd.DataFrame({
            'person_id':[1],
            'cohort_start_date':[
                pd.to_datetime('2017-12-10')
            ]
        })
        pd.testing.assert_frame_equal(output, expected)


class TestFeaturesSelection():
    def test_feature_age_gender(self):
        features = {
            'non_time_bound':{
                'age_group':0,
                'age_at_index':1,
                'gender':1
            },
            'time_bound':{
                'comorbid_condition': [0, 0, 0, 0],
                'drug': [0, 0, 0, 0],
                'condition': [0, 0, 0, 0],
                'procedure': [0, 0, 0, 0],
                'measurement': [0, 0, 0, 0],
                'measurement_value': [0, 0, 0, 0],
                'measurement_range_group': [0, 0, 0, 0],
                'visit_count': [0, 0, 0, 0]
            },
            'time_windows':{
                'inf':5000,
                'long':365,
                'med':180,
                'short':30,
                'minimum':0.05
            }
        }

        expected = pd.DataFrame({
            'person_id':[1, 2, 3, 4, 5],
            'cohort_start_date':[pd.to_datetime('2018-01-01')]*5,
            'age_at_index':[28, 18, 8, 48, 58],
            'gender = female':[1, 0, 1, 0, 0]
        })

        output = FeaturesSelection(cohort, features,
                    drug_exposure, condition_occurrence, visit_occurrence, person, measurement, procedure)()

        pd.testing.assert_frame_equal(output, expected)
        

    def test_feature_age_group_minimum(self):
        features = {
            'non_time_bound':{
                'age_group': 1,
                'age_at_index': 0,
                'gender': 0
            },
            'time_bound':{
                'comorbid_condition': [0, 0, 0, 0],
                'drug': [0, 0, 0, 0],
                'condition': [0, 0, 0, 0],
                'procedure': [0, 0, 0, 0],
                'measurement': [0, 0, 0, 0],
                'measurement_value': [0, 0, 0, 0],
                'measurement_range_group': [0, 0, 0, 0],
                'visit_count': [0, 0, 0, 0]
            },
            'time_windows':{
                'inf': 5000,
                'long': 365,
                'med': 180,
                'short': 30,
                'minimum': 0.05
            }
        }

        expected = pd.DataFrame({
            'person_id': [1, 2, 3, 4, 5],
            'cohort_start_date': [pd.to_datetime('2018-01-01')]*5,
            '05-09': [0, 0, 1, 0, 0],
            '15-19': [0, 1, 0, 0, 0],
            '25-29': [1, 0, 0, 0, 0],
            '45-49': [0, 0, 0, 1, 0],
            '55-59': [0, 0, 0, 0, 1]
        })

        output = FeaturesSelection(cohort, features,
                    drug_exposure, condition_occurrence, visit_occurrence, person, measurement, procedure)()
        pd.testing.assert_frame_equal(output, expected)
        
        
    def test_feature_conditions(self):
        features = {
            'non_time_bound':{
                'age_group': 0,
                'age_at_index': 0,
                'gender': 0
            },
            'time_bound':{
                'comorbid_condition':[0, 0, 0, 0],
                'drug':[0, 0, 0, 0],
                'condition':[1, 1, 1, 1],
                'procedure': [0, 0, 0, 0],
                'measurement': [0, 0, 0, 0],
                'measurement_value': [0, 0, 0, 0],
                'measurement_range_group': [0, 0, 0, 0],
                'visit_count': [0, 0, 0, 0]
            },
            'time_windows':{
                'inf': 5000,
                'long': 365,
                'med': 180,
                'short': 30,
                'minimum': 0.05
            }
        }

        expected = pd.DataFrame({
            'person_id': [1, 2, 3, 4, 5],
            'cohort_start_date': [pd.to_datetime('2018-01-01')]*5,
            '44831230_inf': [1, 1, 0, 0, 0],
            '2_inf':[1, 1, 0, 0, 0],
            '3_inf':[1, 0, 0, 0, 0],
            '4_inf':[1, 0, 0, 0, 0],
            '44831230_long': [1, 1, 0, 0, 0],
            '2_long':[1, 1, 0, 0, 0],
            '3_long':[1, 0, 0, 0, 0],
            '4_long':[1, 0, 0, 0, 0],
            '44831230_med': [1, 1, 0, 0, 0],
            '2_med':[1, 1, 0, 0, 0],
            '3_med':[1, 0, 0, 0, 0],
            '4_med':[1, 0, 0, 0, 0],
            '44831230_short': [1, 1, 0, 0, 0],
            '2_short':[1, 1, 0, 0, 0],
            '3_short':[1, 0, 0, 0, 0],
            '4_short':[1, 0, 0, 0, 0],
        })

        output = FeaturesSelection(cohort, features,
                    drug_exposure, condition_occurrence, visit_occurrence, person, measurement, procedure)()
        pd.testing.assert_frame_equal(output, expected)
        
        
    def test_feature_visit_count(self):
        features = {
            'non_time_bound':{
                'age_group': 0,
                'age_at_index': 0,
                'gender': 0
            },
            'time_bound':{
                'comorbid_condition': [0, 0, 0, 0],
                'drug': [0, 0, 0, 0],
                'condition': [0, 0, 0, 0],
                'procedure': [0, 0, 0, 0],
                'measurement': [0, 0, 0, 0],
                'measurement_value': [0, 0, 0, 0],
                'measurement_range_group': [0, 0, 0, 0],
                'visit_count': [0, 0, 1, 0]
            },
            'time_windows':{
                'inf': 5000,
                'long': 365,
                'med': 180,
                'short': 30,
                'minimum': 0.05
            }
        }

        expected = pd.DataFrame({
            'person_id':[1, 2, 3, 4, 5],
            'cohort_start_date':[pd.to_datetime('2018-01-01')]*5,
            'visit_count_med':[1, 0, 0, 0, 0]
        })
        output = FeaturesSelection(cohort, features,
                    drug_exposure, condition_occurrence, visit_occurrence, person, measurement, procedure)()
        pd.testing.assert_frame_equal(output, expected)
        
        
    def test_feature_comorbidities(self):
        features = {
            'non_time_bound':{
                'age_group': 0,
                'age_at_index': 0,
                'gender': 0
            },
            'time_bound':{
                'comorbid_condition': [0, 0, 1, 0],
                'drug': [0, 0, 0, 0],
                'condition': [0, 0, 0, 0],
                'procedure': [0, 0, 0, 0],
                'measurement': [0, 0, 0, 0],
                'measurement_value': [0, 0, 0, 0],
                'measurement_range_group': [0, 0, 0, 0],
                'visit_count': [0, 0, 0, 0]
            },
            'time_windows':{
                'inf': 5000,
                'long': 365,
                'med': 180,
                'short': 30,
                'minimum': 0.05
            }
        }

        expected = pd.DataFrame({
            'person_id':[1, 2, 3, 4, 5],
            'cohort_start_date':[pd.to_datetime('2018-01-01')]*5,
            'congestive_heart_failure_med':[0, 0, 1, 0, 0],
            'valvular_disease_med':[0, 0, 0, 1, 0],
            'hypertension,complicated_med':[0, 0, 1, 0, 0]
        })
        condition_occurrence = pd.DataFrame({
            'person_id':[1, 1, 1, 1, 2, 2, 3, 4],
            'condition_concept_id':[1, 2, 3, 4, 1, 2, 44831230, 44836801],
            'condition_start_datetime':[
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10')
            ]
        })
        condition_occurrence = dd.from_pandas(condition_occurrence, npartitions=1)

        output = FeaturesSelection(cohort, features,
                    drug_exposure, condition_occurrence, visit_occurrence, person, measurement, procedure)()
        pd.testing.assert_frame_equal(output, expected)