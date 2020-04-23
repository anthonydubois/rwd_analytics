import pandas as pd
import dask.dataframe as dd
import pytest

from rwd_analytics.cohort import CohortBuilder, get_distribution, Cohort
from rwd_analytics.treatment_line import EraCalculation, last_activity_date, line_generation_preprocess, LinesOfTherapy, LineName, agg_lot_by_patient
from rwd_analytics.features_selection import FeaturesSelection, time_at_risk, get_features_scores
from rwd_analytics.lookups import Descendants, Concept, ConceptRelationship, ComorbidConditions, Ingredient
from rwd_analytics.predictions import get_matching_pairs
from rwd_analytics.data_cleaning import CleaningLabResults


class TestDataCleaning():
    def test_cleaning_lab_results(self):
        df = pd.DataFrame({
            'person_id':[1, 1, 1, 1, 1, 1, 1],
            'measurement_concept_id':[100, 100, 100, 100, 100, 200, 200],
            'value_as_number':[2, 4, 6, 3000, 0.6, 19800, 0.192],
            'range_high':[15, 15, 0, 15, 15, 210, 0],
            'range_low':[5, 5, 5, 5, 5, 120, 0]
        })
        output = CleaningLabResults(df)()
        expected = pd.DataFrame({
            'person_id':[1, 1, 1, 1, 1, 1, 1],
            'measurement_concept_id':[100, 100, 100, 100, 100, 200, 200],
            'value_as_number':[2.0, 4.0, 6.0, 3.0, 6.0, 198.0, 192.0],
            'range_high':[15.0, 15.0, 15.0, 15.0, 15.0, 210.0, 210.0],
            'range_low':[5.0, 5.0, 5.0, 5.0, 5.0, 120.0, 120.0]
        })
        pd.testing.assert_frame_equal(output, expected)


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
omop_tables = {
    'person':person,
    'condition_occurrence':condition_occurrence,
    'procedure_occurrence':procedure,
    'drug_exposure':drug_exposure,
    'visit_occurrence':visit_occurrence,
    'observation_period':observation_period,
    'measurement':measurement
}


class TestCohort():
    def test_cohort_information(self):
        cohort = pd.DataFrame({
            'person_id':[1, 2, 3],
            'index_date':[
                pd.to_datetime('2015-10-01'),
                pd.to_datetime('2015-10-01'),
                pd.to_datetime('2015-10-01')
            ]
        })
        information = Cohort(omop_tables, cohort).demographics()
        output_age = information['age_at_index']
        output_gender = information['gender_concept_id']
        output_index = information['index_date']
        expected_age = pd.DataFrame({
            'age_at_index':[15, 5, 25],
            'count':[1]*3
        })
        expected_gender = pd.DataFrame({
            'gender_concept_id':['Female', 'Male'],
            'count':[2, 1]
        })
        expected_index = pd.DataFrame({
            'index_date':[2015],
            'count':[3]
        })
        pd.testing.assert_frame_equal(output_age, expected_age)
        pd.testing.assert_frame_equal(output_gender, expected_gender)
        pd.testing.assert_frame_equal(output_index, expected_index)


class TestCohortBuilder():
    def test_distribution_standard(self):
        drug_exposure = pd.DataFrame({
            'person_id':[1, 1, 1, 1, 2, 2],
            'drug_source_concept_id':[1510703, 1125315, 1125315, 40, 1125315, 1510703],
            'drug_concept_id':[1510703, 20, 1125315, 40, 1125315, 20],
            'drug_exposure_start_datetime':[
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-12'),
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10'),
            ]
        })
        drug_exposure = dd.from_pandas(drug_exposure, npartitions=1).set_index('person_id')
        omop_tables = {
            'person':person,
            'condition_occurrence':condition_occurrence,
            'procedure_occurrence':procedure,
            'drug_exposure':drug_exposure,
            'visit_occurrence':visit_occurrence,
            'observation_period':observation_period,
            'measurement':measurement
        }
        output=get_distribution(omop_tables, [1510703, 1125315], '2017-12-09',
                                '2017-12-11', cohort=None)
        expected = pd.DataFrame({
            'concept_id':[1125315, 1510703],
            'concept_code':['161', '2047647'],
            'vocabulary_id':['RxNorm']*2,
            'concept_name':['Acetaminophen', 'Helleborus extract'],
            'n_unique_patients':[1, 1],
            'n_records':[1, 1]
        })
        pd.testing.assert_frame_equal(output, expected)

    def test_distribution_source(self):
        drug_exposure = pd.DataFrame({
            'person_id':[1, 1, 1, 1, 2, 2],
            'drug_source_concept_id':[1510703, 1125315, 1125315, 40, 1125315, 1510703],
            'drug_concept_id':[1510703, 20, 1125315, 40, 1125315, 20],
            'drug_exposure_start_datetime':[
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-12'),
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10'),
            ]
        })
        drug_exposure = dd.from_pandas(drug_exposure, npartitions=1).set_index('person_id')
        omop_tables = {
            'person':person,
            'condition_occurrence':condition_occurrence,
            'procedure_occurrence':procedure,
            'drug_exposure':drug_exposure,
            'visit_occurrence':visit_occurrence,
            'observation_period':observation_period,
            'measurement':measurement
        }
        concept_ids_ingredients = [1510703, 1125315]
        concept_ids = Descendants()(concept_ids_ingredients)
        output = get_distribution(omop_tables, concept_ids, start_date=None,
                                end_date=None, cohort=None)
        expected = pd.DataFrame({
            'concept_id':[1125315, 1510703],
            'concept_code':['161', '2047647'],
            'vocabulary_id':['RxNorm']*2,
            'concept_name':['Acetaminophen', 'Helleborus extract'],
            'n_unique_patients':[2, 1],
            'n_records':[2, 1]
        })
        pd.testing.assert_frame_equal(output, expected)
        
    def test_distribution_cohort_filter(self):
        drug_exposure = pd.DataFrame({
            'person_id':[1, 1, 1, 1, 2, 2],
            'drug_source_concept_id':[1510703, 1125315, 1125315, 40, 1125315, 1510703],
            'drug_concept_id':[1510703, 20, 1125315, 40, 1125315, 20],
            'drug_exposure_start_datetime':[
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-12'),
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10'),
            ]
        })
        drug_exposure = dd.from_pandas(drug_exposure, npartitions=1).set_index('person_id')
        omop_tables = {
            'person':person,
            'condition_occurrence':condition_occurrence,
            'procedure_occurrence':procedure,
            'drug_exposure':drug_exposure,
            'visit_occurrence':visit_occurrence,
            'observation_period':observation_period,
            'measurement':measurement
        }
        cohort = pd.DataFrame({
            'person_id':[1],
            'index_date':[pd.to_datetime('2016-01-01')]
        })
        output = get_distribution(omop_tables, [1510703, 1125315], cohort=cohort)
        expected = pd.DataFrame({
            'concept_id':[1125315, 1510703],
            'concept_code':['161', '2047647'],
            'vocabulary_id':['RxNorm']*2,
            'concept_name':['Acetaminophen', 'Helleborus extract'],
            'n_unique_patients':[1, 1],
            'n_records':[1, 1]
        })
        pd.testing.assert_frame_equal(output, expected)

    def test_gender(self):
        cohort_definition = {
            'demographic_criteria':{
                'min_year_of_birth':None,
                'max_year_of_birth':None,
                'gender_concept_id':[8507]
            },
            'criteria_list':[
            ],
            'observation_period_criteria':{
                'before_index':None,
                'after_index':None
            }
        }
        output = CohortBuilder(cohort_definition, omop_tables)()
        output = output.reset_index(drop=True)
        expected = pd.DataFrame({
            'person_id':[2, 4, 5],
            'index_date':[
                pd.to_datetime('1970-01-01'),
                pd.to_datetime('1970-01-01'),
                pd.to_datetime('1970-01-01')
            ]
        })
        pd.testing.assert_frame_equal(output, expected)
        
    def test_condition(self):
        cohort_definition = {
            'demographic_criteria':{
                'min_year_of_birth':None,
                'max_year_of_birth':None,
                'gender_concept_id':None
            },
            'criteria_list':[
                {
                    'concept_type':'condition_occurrence',
                    'concept_id':[3],
                    'is_excluded':0,
                    'get_descendants':0,
                    'occurrence_start_date':None,
                    'occurrence_end_date':None,
                    'min_occurrence':None,
                    'min_duration':None,
                    'is_before_previous_criteria':None,
                    'is_after_previous_criteria':None
                }
            ],
            'observation_period_criteria':{
                'before_index':None,
                'after_index':None
            }
        }
        output = CohortBuilder(cohort_definition, omop_tables)()
        expected = pd.DataFrame({
            'person_id':[1],
            'index_date':[
                pd.to_datetime('2017-12-10')
            ]
        })
        pd.testing.assert_frame_equal(output, expected)
        
        
    def test_multiple_criteria(self):
        cohort_definition = {
            'demographic_criteria':{
                'min_year_of_birth':None,
                'max_year_of_birth':None,
                'gender_concept_id':[8507]
            },
            'criteria_list':[
                {
                    'concept_type':'condition_occurrence',
                    'concept_id':[44831230],
                    'is_excluded':0,
                    'get_descendants':0,
                    'occurrence_start_date':None,
                    'occurrence_end_date':None,
                    'min_occurrence':None,
                    'min_duration':None,
                    'is_before_previous_criteria':None,
                    'is_after_previous_criteria':None
                }
            ],
            'observation_period_criteria':{
                'before_index':None,
                'after_index':None
            }
        }
        output = CohortBuilder(cohort_definition, omop_tables)()
        expected = pd.DataFrame({
            'person_id':[2],
            'index_date':[
                pd.to_datetime('2017-12-10')
            ]
        })
        pd.testing.assert_frame_equal(output, expected)

    def test_year_of_birth(self):
        cohort_definition = {
            'demographic_criteria':{
                'min_year_of_birth':1970,
                'max_year_of_birth':2000,
                'gender_concept_id':None
            },
            'criteria_list':[
            ],
            'observation_period_criteria':{
                'before_index':None,
                'after_index':None
            }
        }
        output = CohortBuilder(cohort_definition, omop_tables)()
        output = output.reset_index(drop=True)
        expected = pd.DataFrame({
            'person_id':[1, 2, 4],
            'index_date':[
                pd.to_datetime('1970-01-01'),
                pd.to_datetime('1970-01-01'),
                pd.to_datetime('1970-01-01')
            ]
        })
        pd.testing.assert_frame_equal(output, expected)

    def test_previous_cohort(self):
        cohort_definition = {
            'demographic_criteria':{
                'min_year_of_birth':None,
                'max_year_of_birth':None,
                'gender_concept_id':None
            },
            'criteria_list':[
                {
                    'concept_type':'drug_exposure',
                    'concept_id':[10],
                    'is_excluded':0,
                    'get_descendants':0,
                    'occurrence_start_date':None,
                    'occurrence_end_date':None,
                    'min_occurrence':None,
                    'min_duration':None,
                    'is_before_previous_criteria':None,
                    'is_after_previous_criteria':None
                }
            ],
            'observation_period_criteria':{
                'before_index':None,
                'after_index':None
            }
        }
        cohort = pd.DataFrame({
            'person_id':[1],
            'index_date':[pd.to_datetime('1990-01-01', format='%Y-%m-%d')]
        })
        output = CohortBuilder(cohort_definition, omop_tables, cohort)()
        output = output.reset_index(drop=True)
        expected = pd.DataFrame({
            'person_id':[1],
            'index_date':[
                pd.to_datetime('2017-12-10')
            ]
        })
        pd.testing.assert_frame_equal(output, expected)

    def test_observation_period(self):
        cohort_definition = {
            'demographic_criteria':{
                'min_year_of_birth':None,
                'max_year_of_birth':None,
                'gender_concept_id':None
            },
            'criteria_list':[
                {
                    'concept_type':'condition_occurrence',
                    'concept_id':[44831230],
                    'is_excluded':0,
                    'get_descendants':0,
                    'occurrence_start_date':None,
                    'occurrence_end_date':None,
                    'min_occurrence':None,
                    'min_duration':None,
                    'is_before_previous_criteria':None,
                    'is_after_previous_criteria':None
                }
            ],
            'observation_period_criteria':{
                'before_index':365,
                'after_index':None
            }
        }
        output = CohortBuilder(cohort_definition, omop_tables)()
        output = output.reset_index(drop=True)
        expected = pd.DataFrame({
            'person_id':[1],
            'index_date':[
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
        cohort_definition = {
            'demographic_criteria':{
                'min_year_of_birth':None,
                'max_year_of_birth':None,
                'gender_concept_id':None
            },
            'criteria_list':[
                {
                    'concept_type':'drug_exposure',
                    'concept_id':[10],
                    'is_excluded':0,
                    'get_descendants':0,
                    'occurrence_start_date':None,
                    'occurrence_end_date':None,
                    'min_occurrence':2,
                    'min_duration':None,
                    'is_before_previous_criteria':None,
                    'is_after_previous_criteria':None
                }
            ],
            'observation_period_criteria':{
                'before_index':None,
                'after_index':None
            }
        }
        omop_tables = {
            'person':person,
            'condition_occurrence':condition_occurrence,
            'procedure_occurrence':procedure,
            'drug_exposure':drug_exposure,
            'visit_occurrence':visit_occurrence,
            'observation_period':observation_period,
            'measurement':measurement
        }
        output = CohortBuilder(cohort_definition, omop_tables)()
        expected = pd.DataFrame({
            'person_id':[1],
            'index_date':[pd.to_datetime('2017-12-10', format = '%Y-%m-%d')]
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
        
        cohort_definition = {
            'demographic_criteria':{
                'min_year_of_birth':None,
                'max_year_of_birth':None,
                'gender_concept_id':None
            },
            'criteria_list':[
                {
                    'concept_type':'condition_occurrence',
                    'concept_id':[44831230],
                    'is_excluded':0,
                    'get_descendants':0,
                    'occurrence_start_date':None,
                    'occurrence_end_date':None,
                    'min_occurrence':None,
                    'min_duration':None,
                    'is_before_previous_criteria':None,
                    'is_after_previous_criteria':None
                },
                {
                    'concept_type':'drug_exposure',
                    'concept_id':[10],
                    'is_excluded':0,
                    'get_descendants':0,
                    'occurrence_start_date':None,
                    'occurrence_end_date':None,
                    'min_occurrence':None,
                    'min_duration':None,
                    'is_before_previous_criteria':1,
                    'is_after_previous_criteria':None
                }
            ],
            'observation_period_criteria':{
                'before_index':None,
                'after_index':None
            }
        }

        omop_tables = {
            'person':person,
            'condition_occurrence':condition_occurrence,
            'procedure_occurrence':procedure,
            'drug_exposure':drug_exposure,
            'visit_occurrence':visit_occurrence,
            'observation_period':observation_period,
            'measurement':measurement
        }
        output = CohortBuilder(cohort_definition, omop_tables)()
        expected = pd.DataFrame({
            'person_id':[1],
            'index_date':[pd.to_datetime('2018-12-10')]
        })
        pd.testing.assert_frame_equal(output, expected)

    def test_attributes_min_length(self):
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
        cohort_definition = {
            'demographic_criteria':{
                'min_year_of_birth':None,
                'max_year_of_birth':None,
                'gender_concept_id':None
            },
            'criteria_list':[
                {
                    'concept_type':'drug_exposure',
                    'concept_id':[10],
                    'is_excluded':0,
                    'get_descendants':0,
                    'occurrence_start_date':None,
                    'occurrence_end_date':None,
                    'min_occurrence':None,
                    'min_duration':100,
                    'is_before_previous_criteria':None,
                    'is_after_previous_criteria':None
                }
            ],
            'observation_period_criteria':{
                'before_index':None,
                'after_index':None
            }
        }

        omop_tables = {
            'person':person,
            'condition_occurrence':condition_occurrence,
            'procedure_occurrence':procedure,
            'drug_exposure':drug_exposure,
            'visit_occurrence':visit_occurrence,
            'observation_period':observation_period,
            'measurement':measurement
        }
        output = CohortBuilder(cohort_definition, omop_tables)()
        expected = pd.DataFrame({
            'person_id':[1],
            'index_date':[pd.to_datetime('2017-12-10')]
        })
        pd.testing.assert_frame_equal(output, expected)


class TestFeaturesSelection():
    def test_feature_age_gender(self):
        cohort = pd.DataFrame({
            'person_id':[1, 2, 3, 4, 5, 6],
            'index_date':[
                pd.to_datetime('2018-01-01'),
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
            'index_date':[pd.to_datetime('2018-01-01')]*5,
            'age_at_index':[28, 18, 8, 48, 58],
            'gender = female':[1, 0, 1, 0, 0]
        })

        output = FeaturesSelection(cohort, features, omop_tables)()

        pd.testing.assert_frame_equal(output, expected)
        

    def test_feature_age_group_minimum(self):
        cohort = pd.DataFrame({
            'person_id':[1, 2, 3, 4, 5],
            'index_date':[
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
            'index_date': [pd.to_datetime('2018-01-01')]*5,
            '05-09': [0, 0, 1, 0, 0],
            '15-19': [0, 1, 0, 0, 0],
            '25-29': [1, 0, 0, 0, 0],
            '45-49': [0, 0, 0, 1, 0],
            '55-59': [0, 0, 0, 0, 1]
        })

        output = FeaturesSelection(cohort, features, omop_tables)()
        pd.testing.assert_frame_equal(output, expected)
        
        
    def test_feature_conditions(self):
        cohort = pd.DataFrame({
            'person_id':[1, 2, 3, 4, 5],
            'index_date':[
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
            'index_date': [pd.to_datetime('2018-01-01')]*5,
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

        output = FeaturesSelection(cohort, features, omop_tables)()
        pd.testing.assert_frame_equal(output, expected)
        
        
    def test_feature_visit_count(self):
        cohort = pd.DataFrame({
            'person_id':[1, 2, 3, 4, 5],
            'index_date':[
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
            'index_date':[pd.to_datetime('2018-01-01')]*5,
            'visit_count_med':[1, 0, 0, 0, 0]
        })
        output = FeaturesSelection(cohort, features, omop_tables)()
        pd.testing.assert_frame_equal(output, expected)
        
        
    def test_feature_comorbidities(self):
        cohort = pd.DataFrame({
            'person_id':[1, 2, 3, 4, 5],
            'index_date':[
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
            'index_date':[pd.to_datetime('2018-01-01')]*5,
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
        omop_tables = {
            'person':person,
            'condition_occurrence':condition_occurrence,
            'procedure_occurrence':procedure,
            'drug_exposure':drug_exposure,
            'visit_occurrence':visit_occurrence,
            'observation_period':observation_period,
            'measurement':measurement
        }
        output = FeaturesSelection(cohort, features, omop_tables)()
        pd.testing.assert_frame_equal(output, expected)


def test_time_at_risk():
    cohort = pd.DataFrame({
        'cohort_definition_id':[1, 1, 1, 1, 1, 2, 2, 2],
        'person_id':[1, 2, 3, 4, 5, 1, 2, 3],
        'index_date':[
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
    X = FeaturesSelection(cohort_at_risk, features, omop_tables)()
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
        omop_tables = {
            'person':person,
            'condition_occurrence':condition_occurrence,
            'procedure_occurrence':procedure,
            'drug_exposure':drug_exposure,
            'visit_occurrence':visit_occurrence,
            'observation_period':observation_period,
            'measurement':measurement
        }
        output = last_activity_date(cohort, omop_tables)
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
            'index_date':[
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
            'index_date':[
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
        output = Concept().get_info(df, ['concept_name', 'domain_id'])
        expected = pd.DataFrame({
            'person_id':[1, 2, 3],
            'concept_id':[43012292, 43012292, 43012292],
            'concept_name':['cabozantinib']*3,
            'domain_id':['Drug']*3
        })
        pd.testing.assert_frame_equal(output, expected)

    def test_search(self):
        output = Concept()
        output = output.search_for_concept_by_name('cabozantinib 40 MG Oral Tablet')
        output = output[['concept_id', 'concept_name']].reset_index(drop=True)
        expected = pd.DataFrame({
            'concept_id':[42629389, 42629391],
            'concept_name':['cabozantinib 40 MG Oral Tablet', 'cabozantinib 40 MG Oral Tablet [Cabometyx]']
        })
        pd.testing.assert_frame_equal(output, expected)

    def test_get_concept_name(self):
        c = Concept()
        output = c.get_unique_concept_name(35603073)
        expected = 'irinotecan hydrochloride liposome 4.3 MG/ML [Onivyde]'
        assert output == expected
        
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


class TestTreatmentLine():
    def test_line_generation(self):
        cohort = pd.DataFrame({
            'person_id':[1, 2],
            'index_date':[
                pd.to_datetime('2017-11-17', format='%Y-%m-%d'),
                pd.to_datetime('2017-11-17', format='%Y-%m-%d')
            ]
        })
        drug_exposure = pd.DataFrame({
            'person_id':[1, 1, 1, 1, 2, 2, 2],
            'drug_concept_id':[46276410, 46276410, 46276410, 46276410, 46276410, 46276410, 25],
            'drug_exposure_start_datetime':[
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10')
            ]
        })
        drug_exposure = dd.from_pandas(drug_exposure, npartitions=1).set_index('person_id')
        omop_tables = {
            'person':person,
            'condition_occurrence':condition_occurrence,
            'procedure_occurrence':procedure,
            'drug_exposure':drug_exposure,
            'visit_occurrence':visit_occurrence,
            'observation_period':observation_period,
            'measurement':measurement
        }
        ingredient_list = [45775965]
        drug_temp, cohort_enhanced = line_generation_preprocess(cohort, ingredient_list, omop_tables)
        output = LinesOfTherapy(drug_temp, cohort_enhanced, ingredient_list, nb_of_lines=1)()
        expected = pd.DataFrame({
            'person_id':[1, 2],
            'line_number':[1]*2,
            'regimen_name':['pembrolizumab']*2,
            'start_date':[
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10')
            ],
            'end_date':[
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10')
            ]
        })
        pd.testing.assert_frame_equal(output, expected)

    def test_lot_generation_2(self):
        cohort = pd.DataFrame({
            'person_id':[1, 2, 3],
            'index_date':[
                pd.to_datetime('2017-01-10', format='%Y-%m-%d'),
                pd.to_datetime('2017-11-17', format='%Y-%m-%d'),
                pd.to_datetime('2019-01-01', format='%Y-%m-%d')
            ]
        })
        drug_exposure = pd.DataFrame({
            'person_id':[1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3],
            'drug_concept_id':[46276410, 46276410, 955632, 955632, 1337620, 40168303, 1367268,
                            1314924, 1378382,
                            1314924, 1378382],
            'drug_exposure_start_datetime':[
                pd.to_datetime('2017-01-01'),
                pd.to_datetime('2017-02-10'),
                pd.to_datetime('2017-01-27'),
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-01-31'),
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-10-10'),
                pd.to_datetime('2018-01-01'),
                pd.to_datetime('2018-02-28'),
                pd.to_datetime('2019-01-01'),
                pd.to_datetime('2019-06-30')
            ]
        })
        drug_exposure = dd.from_pandas(drug_exposure, npartitions=1).set_index('person_id')
        omop_tables = {
            'person':person,
            'condition_occurrence':condition_occurrence,
            'procedure_occurrence':procedure,
            'drug_exposure':drug_exposure,
            'visit_occurrence':visit_occurrence,
            'observation_period':observation_period,
            'measurement':measurement
        }
        ingredient_list = [46276410, 45775965, 955632, 1337620, 40168303, 1367268, 1314924, 1378382]
        drug_temp, cohort_enhanced = line_generation_preprocess(cohort, ingredient_list, omop_tables)
        output = LinesOfTherapy(drug_temp, cohort_enhanced, ingredient_list, nb_of_lines=4)()
        expected = """
            {
                "person_id":{"0":1,"1":1,"2":1,"3":2,"4":3,"5":3},
                "line_number":{"0":1,"1":2,"2":3,"3":1,"4":1,"5":2},
                "regimen_name":{"0":"Fluorouracil, pembrolizumab","1":"irinotecan","2":"Fluorouracil, levoleucovorin","3":"Paclitaxel, gemcitabine","4":"gemcitabine","5":"Paclitaxel"},
                "start_date":{"0":1483228800000,"1":1507593600000,"2":1512864000000,"3":1514764800000,"4":1546300800000,"5":1561852800000},
                "end_date":{"0":1507507200000,"1":1512777600000,"2":1512864000000,"3":1519776000000,"4":1561766400000,"5":1561852800000}
            }
        """
        expected = pd.read_json(expected)
        expected['start_date'] = pd.to_datetime(expected['start_date'], unit='ms')
        expected['end_date'] = pd.to_datetime(expected['end_date'], unit='ms')
        pd.testing.assert_frame_equal(output, expected)

    def test_line_agg_by_patient(self):
        lot = pd.DataFrame({
            'person_id':[1, 1],
            'start_date':[
                pd.to_datetime('2017-11-10'),
                pd.to_datetime('2017-12-10')
            ],
            'line_number':[1, 2],
            'end_date':[
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2018-04-10')
            ],
            'regimen_name':['pembrolizumab', 'cabozantinib']
        })
        output = agg_lot_by_patient(lot)
        expected = pd.DataFrame({
            'person_id':[1],
            'start_date 1L':[pd.to_datetime('2017-11-10')],
            'end_date 1L':[pd.to_datetime('2017-12-10')],
            'regimen 1L':['pembrolizumab'],
            'start_date 2L':[pd.to_datetime('2017-12-10')],
            'end_date 2L':[pd.to_datetime('2018-04-10')],
            'regimen 2L':['cabozantinib'],
            'start_date 3L':[pd.NaT],
            'end_date 3L':[pd.NaT],
            'regimen 3L':['no 3L']
        })
        pd.testing.assert_frame_equal(output, expected)


class TestPrediction():
    def test_matching_pairs(self):
        cohort_plus = pd.DataFrame({
            'person_id':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'age_at_index':[8, 18, 28, 48, 58, 38, 40, 38, 46, 50],
            'gender = female':[1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
            'diabete':[0, 1, 0, 1, 0, 1, 1, 1, 0, 0],
            'doliprane':[1, 1, 0, 0, 1, 0, 0, 1, 1, 0],
            'target':[1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        })
        treated_df = cohort_plus[cohort_plus['target']==1]
        del treated_df['target']

        new_patient = cohort_plus[cohort_plus['target']==0]
        del new_patient['target']
        output = get_matching_pairs(treated_df, new_patient, distance_max = 2.5)
        expected = pd.DataFrame({
            'person_id':[9, 5, 6],
            'age_at_index':[46, 58, 38],
            'gender = female':[0]*3,
            'diabete':[0, 0, 1],
            'doliprane':[1, 1, 0]
        })
        pd.testing.assert_frame_equal(output, expected)