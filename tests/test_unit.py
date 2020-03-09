import pandas as pd
import dask.dataframe as dd
import pytest

from rwd_analytics.cohort import CohortBuilder
from rwd_analytics.treatment_line import EraCalculation, last_activity_date
from rwd_analytics.features_selection import FeaturesSelection, time_at_risk, get_features_scores
from rwd_analytics.lookups import Descendants, ConceptInfo, ConceptRelationship, ComorbidConditions, Ingredient

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
observation_period = pd.DataFrame({
    'person_id':[1, 2],
    'observation_period_start_date':[
        pd.to_datetime('2015-01-01'),
        pd.to_datetime('2017-12-01')
    ],
    'observation_period_end_date':[
        pd.to_datetime('2019-01-01'),
        pd.to_datetime('2018-02-01')
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
visit_occurrence = dd.from_pandas(visit_occurrence, npartitions=1).set_index('person_id')
person = dd.from_pandas(person, npartitions=1).set_index('person_id')
condition_occurrence = dd.from_pandas(condition_occurrence, npartitions=1).set_index('person_id')
drug_exposure = dd.from_pandas(drug_exposure, npartitions=1).set_index('person_id')
observation_period = dd.from_pandas(observation_period, npartitions=1).set_index('person_id')
measurement = pd.DataFrame()
procedure = pd.DataFrame()
measurement = dd.from_pandas(measurement, npartitions=1)
procedure = dd.from_pandas(procedure, npartitions=1)


class TestCohort():
    def test_gender(self):
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
        output = CohortBuilder(cohort_criteria, drug_exposure, condition_occurrence, person, observation_period)()
        output = output.reset_index(drop=True)
        expected = pd.DataFrame({
            'person_id':[2, 4, 5],
            'cohort_start_date':[
                pd.to_datetime('1990-01-01'),
                pd.to_datetime('1990-01-01'),
                pd.to_datetime('1990-01-01')
            ]
        })
        pd.testing.assert_frame_equal(output, expected)

    def test_condition(self):
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
        output = CohortBuilder(cohort_criteria, drug_exposure, condition_occurrence, person, observation_period)()
        expected = pd.DataFrame({
            'person_id':[1],
            'cohort_start_date':[
                pd.to_datetime('2017-12-10')
            ]
        })
        pd.testing.assert_frame_equal(output, expected)

    def test_multiple_criteria(self):
        cohort_criteria = {
            'criteria':[
                {
                    'concept_type':'gender',
                    'concept_id':[8507],
                    'excluded':0,
                    'descendant':0,
                    'mapped':0,
                    'attributes':[]
                },
                {
                    'concept_type':'condition',
                    'concept_id':[44831230],
                    'excluded':0,
                    'descendant':0,
                    'mapped':0,
                    'attributes':[]
                }
            ]
        }
        output = CohortBuilder(cohort_criteria, drug_exposure, condition_occurrence, person, observation_period)()
        expected = pd.DataFrame({
            'person_id':[2],
            'cohort_start_date':[
                pd.to_datetime('2017-12-10')
            ]
        })
        pd.testing.assert_frame_equal(output, expected)

    def test_year_of_birth(self):
        cohort_criteria = {
            'criteria':[
                {
                    'concept_type':'year_of_birth',
                    'concept_id':[],
                    'excluded':0,
                    'descendant':0,
                    'mapped':0,
                    'attributes':[
                        {
                            'type':'inferior or equal',
                            'bound':2000
                        },
                        {
                            'type':'superior or equal',
                            'bound':1970
                        }
                    ]
                }
            ]
        }
        output = CohortBuilder(cohort_criteria, drug_exposure, condition_occurrence, person, observation_period)()
        output = output.reset_index(drop=True)
        expected = pd.DataFrame({
            'person_id':[1, 2, 4],
            'cohort_start_date':[
                pd.to_datetime('1990-01-01'),
                pd.to_datetime('1990-01-01'),
                pd.to_datetime('1990-01-01')
            ]
        })
        pd.testing.assert_frame_equal(output, expected)

    def test_previous_cohort(self):
        cohort_criteria = {
            'criteria':[
                {
                    'concept_type':'drug',
                    'concept_id':[10],
                    'excluded':0,
                    'descendant':0,
                    'mapped':0,
                    'attributes':[]
                }
            ]
        }
        cohort = pd.DataFrame({
            'person_id':[1],
            'cohort_start_date':[pd.to_datetime('1990-01-01', format='%Y-%m-%d')]
        })
        output = CohortBuilder(cohort_criteria, drug_exposure, condition_occurrence, 
                               person, observation_period, cohort)()
        output = output.reset_index(drop=True)
        expected = pd.DataFrame({
            'person_id':[1],
            'cohort_start_date':[
                pd.to_datetime('2017-12-10')
            ]
        })
        pd.testing.assert_frame_equal(output, expected)

    def test_observation_period(self):
        cohort_criteria = {
            'criteria':[
                {
                    'concept_type':'condition',
                    'concept_id':[44831230],
                    'excluded':0,
                    'descendant':0,
                    'mapped':0,
                    'attributes':[]
                }
            ],
            'observation_period':{
                'before_index':365,
                'after_index':0
            }
        }
        output = CohortBuilder(cohort_criteria, drug_exposure, condition_occurrence, 
                               person, observation_period)()
        output = output.reset_index(drop=True)
        expected = pd.DataFrame({
            'person_id':[1],
            'cohort_start_date':[
                pd.to_datetime('2017-12-10')
            ]
        })
        pd.testing.assert_frame_equal(output, expected)

    def test_attributes_occurrence(self):
        drug_exposure = pd.DataFrame({
            'person_id':[1, 1, 1, 1, 2, 2],
            'drug_concept_id':[10, 10, 30, 40, 10, 50],
            'drug_exposure_start_datetime':[
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2018-12-10'),
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10'),
            ]
        })
        drug_exposure = dd.from_pandas(drug_exposure, npartitions=1).set_index('person_id')
        cohort_criteria = {
            'criteria':[
                {
                    'concept_type':'drug',
                    'concept_id':[10], # Irinotecan liposomal
                    'excluded':0,
                    'descendant':0,
                    'mapped':0,
                    'attributes':[
                        {
                            'type':'occurrence',
                            'min':2
                        }
                    ]
                }
            ]
        }
        output = CohortBuilder(cohort_criteria, drug_exposure, condition_occurrence, person, observation_period)()
        expected = pd.DataFrame({
            'person_id':[1],
            'cohort_start_date':[pd.to_datetime('2017-12-10', format = '%Y-%m-%d')]
        })
        pd.testing.assert_frame_equal(output, expected)

    def test_condition_with_drug_before(self):
        condition_occurrence = pd.DataFrame({
            'person_id':[1, 1, 1, 1, 2, 2],
            'condition_concept_id':[44831230, 2, 3, 4, 44831230, 2],
            'condition_start_datetime':[
                pd.to_datetime('2018-12-10'),
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
        condition_occurrence = dd.from_pandas(condition_occurrence, npartitions=1).set_index('person_id')
        drug_exposure = dd.from_pandas(drug_exposure, npartitions=1).set_index('person_id')

        cohort_criteria = {
            'criteria':[
                {
                    'concept_type':'condition',
                    'concept_id':[44831230],
                    'excluded':0,
                    'descendant':0,
                    'mapped':0,
                    'attributes':[]
                },
                {
                    'concept_type':'drug',
                    'concept_id':[10],
                    'excluded':0,
                    'descendant':0,
                    'mapped':0,
                    'attributes':[{
                        'type':'before'
                    }]
                }
            ]
        }
        output = CohortBuilder(cohort_criteria, drug_exposure, condition_occurrence,
                            person, observation_period)()
        expected = pd.DataFrame({
            'person_id':[1],
            'cohort_start_date':[pd.to_datetime('2018-12-10')]
        })
        pd.testing.assert_frame_equal(output, expected)

    def test_attributes_min_lenght(self):
        drug_exposure = pd.DataFrame({
            'person_id':[1, 1, 1, 1, 2, 2],
            'drug_concept_id':[10, 10, 30, 40, 10, 20],
            'drug_exposure_start_datetime':[
                pd.to_datetime('2018-12-10'),
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10'),
            ]
        })
        drug_exposure = dd.from_pandas(drug_exposure, npartitions=1).set_index('person_id')
        cohort_criteria = {
            'criteria':[
                {
                    'concept_type':'drug',
                    'concept_id': [10],
                    'excluded': 0,
                    'descendant':0,
                    'mapped':0,
                    'attributes':[{
                        'type':'length',
                        'min':100
                    }]
                }
            ]
        }
        output = CohortBuilder(cohort_criteria, drug_exposure, condition_occurrence, person, observation_period)()
        expected = pd.DataFrame({
            'person_id':[1],
            'cohort_start_date':[pd.to_datetime('2017-12-10')]
        })
        pd.testing.assert_frame_equal(output, expected)


class TestFeaturesSelection():
    def test_feature_age_gender(self):
        cohort = pd.DataFrame({
            'person_id':[1, 2, 3, 4, 5],
            'cohort_start_date':[
                pd.to_datetime('2018-01-01'),
                pd.to_datetime('2018-01-01'),
                pd.to_datetime('2018-01-01'),
                pd.to_datetime('2018-01-01'),
                pd.to_datetime('2018-01-01')
            ]
        })
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
        cohort = pd.DataFrame({
            'person_id':[1, 2, 3, 4, 5],
            'cohort_start_date':[
                pd.to_datetime('2018-01-01'),
                pd.to_datetime('2018-01-01'),
                pd.to_datetime('2018-01-01'),
                pd.to_datetime('2018-01-01'),
                pd.to_datetime('2018-01-01')
            ]
        })
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
        cohort = pd.DataFrame({
            'person_id':[1, 2, 3, 4, 5],
            'cohort_start_date':[
                pd.to_datetime('2018-01-01'),
                pd.to_datetime('2018-01-01'),
                pd.to_datetime('2018-01-01'),
                pd.to_datetime('2018-01-01'),
                pd.to_datetime('2018-01-01')
            ]
        })
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
        cohort = pd.DataFrame({
            'person_id':[1, 2, 3, 4, 5],
            'cohort_start_date':[
                pd.to_datetime('2018-01-01'),
                pd.to_datetime('2018-01-01'),
                pd.to_datetime('2018-01-01'),
                pd.to_datetime('2018-01-01'),
                pd.to_datetime('2018-01-01')
            ]
        })
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
        cohort = pd.DataFrame({
            'person_id':[1, 2, 3, 4, 5],
            'cohort_start_date':[
                pd.to_datetime('2018-01-01'),
                pd.to_datetime('2018-01-01'),
                pd.to_datetime('2018-01-01'),
                pd.to_datetime('2018-01-01'),
                pd.to_datetime('2018-01-01')
            ]
        })
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
        condition_occurrence = dd.from_pandas(condition_occurrence, npartitions=1).set_index('person_id')

        output = FeaturesSelection(cohort, features,
                    drug_exposure, condition_occurrence, visit_occurrence, person, measurement, procedure)()
        pd.testing.assert_frame_equal(output, expected)


def test_time_at_risk():
    cohort = pd.DataFrame({
        'cohort_definition_id':[1, 1, 1, 1, 1, 2, 2, 2],
        'person_id':[1, 2, 3, 4, 5, 1, 2, 3],
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
    cohort_at_risk = cohort[cohort['cohort_definition_id']==1]
    del cohort_at_risk['cohort_definition_id']
    cohort_target = cohort[cohort['cohort_definition_id']==2]
    del cohort_target['cohort_definition_id']
    X = FeaturesSelection(cohort_at_risk, features,
                        drug_exposure, condition_occurrence, visit_occurrence,
                        person, measurement, procedure)()
    output = time_at_risk(X, cohort_at_risk, cohort_target, time_at_risk = 200)
    expected = pd.DataFrame({
        'age_at_index':[28, 18, 8, 48, 58],
        'gender = female':[1, 0, 1, 0, 0],
        'target':[0, 1, 1, 0, 0]
    })
    pd.testing.assert_frame_equal(output, expected)


def test_get_feature_scores():
    df = pd.DataFrame({
        'age_at_index':[28, 18, 8, 48, 58],
        'gender = female':[1, 0, 1, 0, 0],
        'target':[0, 1, 1, 0, 0]
    })
    output = get_features_scores(df, 2)
    expected = pd.DataFrame({
        'Specs':['age_at_index', 'gender = female'],
        'Score':[37.60, 0.08]
    })
    pd.testing.assert_frame_equal(output, expected)


class TestEraCalculation():
    def test_last_activity_date(self):
        cohort = pd.DataFrame({
            'person_id':[1, 2, 3]
        })
        condition_occurrence = pd.DataFrame({
            'person_id':[1, 1, 1, 1, 2, 2],
            'condition_concept_id':[44831230, 2, 3, 4, 44831230, 2],
            'condition_start_datetime':[
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2019-12-10'),
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
                pd.to_datetime('2018-12-10'),
            ]
        })
        condition_occurrence = dd.from_pandas(condition_occurrence, npartitions=1).set_index('person_id')
        drug_exposure = dd.from_pandas(drug_exposure, npartitions=1).set_index('person_id')
        output = last_activity_date(cohort, drug_exposure, condition_occurrence)
        expected = pd.DataFrame({
            'person_id':[1, 2],
            'last_activity_date':[
                pd.to_datetime('2019-12-10'),
                pd.to_datetime('2018-12-10'),
            ]
        })
        pd.testing.assert_frame_equal(output, expected)

    def test_era_without_concept(self):
        cohort = pd.DataFrame({
            'person_id':[1, 2, 3],
            'cohort_start_date':[
                pd.to_datetime('2018-01-01'),
                pd.to_datetime('2018-01-01'),
                pd.to_datetime('2018-01-01')
            ]
        })
        drug_exposure = pd.DataFrame({
            'person_id':[1, 1, 1, 1],
            'drug_concept_id':[10, 10, 10, 20],
            'drug_exposure_start_datetime':[
                pd.to_datetime('2016-01-01'),
                pd.to_datetime('2017-01-01'),
                pd.to_datetime('2018-01-01'),
                pd.to_datetime('2018-01-01')
            ]
        })
        drug_exposure = dd.from_pandas(drug_exposure, npartitions=1)
        drug_exposure = drug_exposure.set_index('person_id')
        output = EraCalculation(cohort, drug_exposure, concept_ids=None)()
        expected = pd.DataFrame({
            'person_id':[1, 1],
            'concept_id':[10, 20],
            'start_date_min':[
                pd.to_datetime('2016-01-01'),
                pd.to_datetime('2018-01-01')
            ],
            'start_date_max':[
                pd.to_datetime('2018-01-01'),
                pd.to_datetime('2018-01-01')
            ],
            'count_exposure':[3, 1],
            'gaps_count':[2, 0],
            'era_duration':[731, 0]
        })
        pd.testing.assert_frame_equal(output, expected)
        
    def test_era_with_concept(self):
        cohort = pd.DataFrame({
            'person_id':[1, 2, 3],
            'cohort_start_date':[
                pd.to_datetime('2018-01-01'),
                pd.to_datetime('2018-01-01'),
                pd.to_datetime('2018-01-01')
            ]
        })
        drug_exposure = pd.DataFrame({
            'person_id':[1, 1, 1, 1],
            'drug_concept_id':[10, 10, 10, 20],
            'drug_exposure_start_datetime':[
                pd.to_datetime('2016-01-01'),
                pd.to_datetime('2017-01-01'),
                pd.to_datetime('2018-01-01'),
                pd.to_datetime('2018-01-01')
            ]
        })
        drug_exposure = dd.from_pandas(drug_exposure, npartitions=1)
        drug_exposure = drug_exposure.set_index('person_id')
        output = EraCalculation(cohort, drug_exposure, concept_ids=[10])()
        expected = pd.DataFrame({
            'person_id':[1],
            'concept_id':[10],
            'start_date_min':[
                pd.to_datetime('2016-01-01')
            ],
            'start_date_max':[
                pd.to_datetime('2018-01-01')
            ],
            'count_exposure':[3],
            'gaps_count':[2],
            'era_duration':[731]
        })
        pd.testing.assert_frame_equal(output, expected)


class TestLookups():
    def test_descendants(self):
        output = Descendants()([43012292])
        expected = [43012038, 43011640, 42629394, 43011643, 43011641,
                    36249906, 42629383, 42629389, 42629387, 42629382,
                    36246764, 42629386, 36246765, 43012292, 43012022,
                    43013204, 43012062, 42629388, 42629385, 43012021,
                    43011642, 43011637, 35805797, 35806421, 43012020,
                    36246763, 42629392, 43012063, 43013205, 36246766,
                    43012064, 42629393, 43011638, 42629390, 43012023,
                    42629391, 36249905, 42629395]
        assert output == expected
        
    def test_concept_info(self):
        df = pd.DataFrame({
            'person_id':[1, 2, 3],
            'concept_id':[43012292, 43012292, 43012292]
        })
        output = ConceptInfo()(df, ['concept_name', 'domain_id'])
        expected = pd.DataFrame({
            'person_id':[1, 2, 3],
            'concept_id':[43012292, 43012292, 43012292],
            'concept_name':['cabozantinib']*3,
            'domain_id':['Drug']*3
        })
        pd.testing.assert_frame_equal(output, expected)
        
    def test_ingredient(self):
        df = pd.DataFrame({
            'person_id':[1, 2, 3],
            'drug_concept_id':[43011640, 42629394, 43011643]
        })
        output = Ingredient()(df)
        expected = pd.DataFrame({
            'person_id':[1, 2, 3],
            'drug_concept_id':[43012292]*3
        })
        pd.testing.assert_frame_equal(output, expected)
        
    def test_comorbid_conditions(self):
        output = ComorbidConditions()('AIDS/H1V')
        expected = ['45552213', '45542556', '45537755', '45566541', '45585950', '45600449',
        '45571459', '45542557', '45556930', '45556931', '45571460', '45571461',
        '45556932', '45571462', '45581152', '45561747', '45532867', '45595610',
        '45547443']
        assert output == expected