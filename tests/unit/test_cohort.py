import pandas as pd
import dask.dataframe as dd

from rwd_analytics.cohort import Cohort, get_distribution
from rwd_analytics.lookups import Descendants


def get_person():
    df = pd.DataFrame({
        'person_id': [1, 2, 3, 4, 5],
        'gender_concept_id': [8532, 8507, 8532, 8507, 8507],
        'year_of_birth': [1990, 2000, 2010, 1970, 1960]
    })
    df = dd.from_pandas(df, npartitions=1).set_index('person_id')
    return df


def get_condition_occurrence():
    df = pd.DataFrame({
        'person_id': [1, 1, 1, 1, 2, 2],
        'condition_concept_id': [44831230, 2, 3, 4, 44831230, 2],
        'condition_start_datetime': [
            pd.to_datetime('2017-12-10'),
            pd.to_datetime('2017-12-10'),
            pd.to_datetime('2017-12-10'),
            pd.to_datetime('2017-12-10'),
            pd.to_datetime('2017-12-10'),
            pd.to_datetime('2017-12-10'),
        ]
    })
    df = dd.from_pandas(df, npartitions=1).set_index('person_id')
    return df


def get_observation_period():
    df = pd.DataFrame({
        'person_id': [1, 2],
        'observation_period_start_date': [
            pd.to_datetime('2015-01-01'),
            pd.to_datetime('2017-12-01')
        ],
        'observation_period_end_date': [
            pd.to_datetime('2019-01-01'),
            pd.to_datetime('2018-02-01')
        ]
    })
    df = dd.from_pandas(df, npartitions=1).set_index('person_id')
    return df


def get_drug_exposure():
    df = pd.DataFrame({
        'person_id': [1, 1, 1, 1, 2, 2],
        'drug_concept_id': [10, 20, 30, 40, 10, 20],
        'drug_exposure_start_datetime': [
            pd.to_datetime('2017-12-10'),
            pd.to_datetime('2017-12-10'),
            pd.to_datetime('2017-12-10'),
            pd.to_datetime('2017-12-10'),
            pd.to_datetime('2017-12-10'),
            pd.to_datetime('2017-12-10'),
        ]
    })
    df = dd.from_pandas(df, npartitions=1).set_index('person_id')
    return df


def get_visit_occurrence():
    df = pd.DataFrame({
        'person_id': [1],
        'visit_start_datetime': [
            pd.to_datetime('2017-12-10')
        ]
    })
    df = dd.from_pandas(df, npartitions=1).set_index('person_id')
    return df


measurement = pd.DataFrame()
procedure = pd.DataFrame()
measurement = dd.from_pandas(measurement, npartitions=1)
procedure = dd.from_pandas(procedure, npartitions=1)


class TestCohort():
    def test_cohort_information(self):
        cohort = pd.DataFrame({
            'person_id': [1, 2, 3],
            'index_date': ['2015-10-01']*3
        })
        cohort['index_date'] = pd.to_datetime(cohort['index_date'])
        observation_period = pd.DataFrame({
            'person_id': [1, 2, 3],
            'observation_period_start_date': ['2015-01-01', '2014-12-01', '2017-12-01'],
            'observation_period_end_date': ['2016-01-01', '2017-01-01', '2017-12-01'],
        })
        observation_period['observation_period_start_date'] = \
            pd.to_datetime(observation_period['observation_period_start_date'])
        observation_period['observation_period_end_date'] = \
            pd.to_datetime(observation_period['observation_period_end_date'])
        observation_period = dd.from_pandas(observation_period, npartitions=1)
        observation_period = observation_period.set_index('person_id')
        omop_tables = {
            'person': get_person(),
            'condition_occurrence': get_condition_occurrence(),
            'procedure_occurrence': procedure,
            'drug_exposure': get_drug_exposure(),
            'visit_occurrence': get_visit_occurrence(),
            'observation_period': observation_period,
            'measurement': measurement
        }
        output = Cohort(omop_tables, cohort).demographics()
        expected = pd.DataFrame({
            'person_id': [1, 2],
            'index_date': ['2015-10-01']*2,
            'age_at_index': [25, 15],
            'gender_concept_id': ['Female', 'Male'],
            'time_before_index': [273, 304],
            'time_after_index': [92, 458]
        })
        expected['index_date'] = pd.to_datetime(expected['index_date'])
        pd.testing.assert_frame_equal(output, expected)


class TestCohortBuilder():
    def test_distribution_standard(self):
        drug_exposure = pd.DataFrame({
            'person_id': [1, 1, 1, 1, 2, 2],
            'drug_source_concept_id': [1510703, 1125315, 1125315, 40, 1125315, 1510703],
            'drug_concept_id': [1510703, 20, 1125315, 40, 1125315, 20],
            'drug_exposure_start_datetime': [
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
            'person': get_person(),
            'condition_occurrence': get_condition_occurrence(),
            'procedure_occurrence': procedure,
            'drug_exposure': drug_exposure,
            'visit_occurrence': get_visit_occurrence(),
            'observation_period': get_observation_period(),
            'measurement': measurement
        }
        output = get_distribution(omop_tables, [1510703, 1125315], '2017-12-09',
                                  '2017-12-11', cohort=None)
        expected = pd.DataFrame({
            'concept_id': [1125315, 1510703],
            'concept_code': ['161', '2047647'],
            'vocabulary_id': ['RxNorm']*2,
            'concept_name': ['Acetaminophen', 'Helleborus extract'],
            'n_unique_patients': [1, 1],
            'n_records': [1, 1]
        })
        pd.testing.assert_frame_equal(output, expected)

    def test_distribution_source(self):
        drug_exposure = pd.DataFrame({
            'person_id': [1, 1, 1, 1, 2, 2],
            'drug_source_concept_id': [1510703, 1125315, 1125315, 40, 1125315, 1510703],
            'drug_concept_id': [1510703, 20, 1125315, 40, 1125315, 20],
            'drug_exposure_start_datetime': [
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
            'person': get_person(),
            'condition_occurrence': get_condition_occurrence(),
            'procedure_occurrence': procedure,
            'drug_exposure': drug_exposure,
            'visit_occurrence': get_visit_occurrence(),
            'observation_period': get_observation_period(),
            'measurement': measurement
        }
        concept_ids_ingredients = [1510703, 1125315]
        concept_ids = Descendants()(concept_ids_ingredients)
        output = get_distribution(omop_tables, concept_ids, start_date=None,
                                  end_date=None, cohort=None)
        expected = pd.DataFrame({
            'concept_id': [1125315, 1510703],
            'concept_code': ['161', '2047647'],
            'vocabulary_id': ['RxNorm']*2,
            'concept_name': ['Acetaminophen', 'Helleborus extract'],
            'n_unique_patients': [2, 1],
            'n_records': [2, 1]
        })
        pd.testing.assert_frame_equal(output, expected)

    def test_distribution_cohort_filter(self):
        drug_exposure = pd.DataFrame({
            'person_id': [1, 1, 1, 1, 2, 2],
            'drug_source_concept_id': [1510703, 1125315, 1125315, 40, 1125315, 1510703],
            'drug_concept_id': [1510703, 20, 1125315, 40, 1125315, 20],
            'drug_exposure_start_datetime': [
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
            'person': get_person(),
            'condition_occurrence': get_condition_occurrence(),
            'procedure_occurrence': procedure,
            'drug_exposure': drug_exposure,
            'visit_occurrence': get_visit_occurrence(),
            'observation_period': get_observation_period(),
            'measurement': measurement
        }
        cohort = pd.DataFrame({
            'person_id': [1],
            'index_date': [pd.to_datetime('2016-01-01')]
        })
        output = get_distribution(omop_tables, [1510703, 1125315], cohort=cohort)
        expected = pd.DataFrame({
            'concept_id': [1125315, 1510703],
            'concept_code': ['161', '2047647'],
            'vocabulary_id': ['RxNorm']*2,
            'concept_name': ['Acetaminophen', 'Helleborus extract'],
            'n_unique_patients': [1, 1],
            'n_records': [1, 1]
        })
        pd.testing.assert_frame_equal(output, expected)
