import pandas as pd
import dask.dataframe as dd

from rwd_analytics.treatment_line import LinesOfTherapy, \
    agg_lot_by_patient, line_generation_preprocess


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


class TestTreatmentLine():
    def test_line_generation(self):
        cohort = pd.DataFrame({
            'person_id': [1, 2],
            'index_date': [
                pd.to_datetime('2017-11-17', format='%Y-%m-%d'),
                pd.to_datetime('2017-11-17', format='%Y-%m-%d')
            ]
        })
        drug_exposure = pd.DataFrame({
            'person_id': [1, 1, 1, 1, 2, 2, 2],
            'drug_concept_id': [46276410, 46276410, 46276410, 46276410, 46276410, 46276410, 25],
            'drug_exposure_start_datetime': [
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
            'person': get_person(),
            'condition_occurrence': get_condition_occurrence(),
            'procedure_occurrence': procedure,
            'drug_exposure': drug_exposure,
            'visit_occurrence': get_visit_occurrence(),
            'observation_period': get_observation_period(),
            'measurement': measurement
        }
        ingredient_list = [45775965]
        drug_temp, cohort_enhanced = line_generation_preprocess(cohort, ingredient_list,
                                                                omop_tables)
        output = LinesOfTherapy(drug_temp, cohort_enhanced, ingredient_list, nb_of_lines=1)()
        expected = pd.DataFrame({
            'person_id': [1, 2],
            'line_number': [1]*2,
            'regimen_name': ['pembrolizumab']*2,
            'start_date': [
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10')
            ],
            'end_date': [
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2017-12-10')
            ]
        })
        pd.testing.assert_frame_equal(output, expected)

    def test_lot_generation_2(self):
        cohort = pd.DataFrame({
            'person_id': [1, 2, 3],
            'index_date': [
                pd.to_datetime('2017-01-10', format='%Y-%m-%d'),
                pd.to_datetime('2017-11-17', format='%Y-%m-%d'),
                pd.to_datetime('2019-01-01', format='%Y-%m-%d')
            ]
        })
        drug_exposure = pd.DataFrame({
            'person_id': [1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3],
            'drug_concept_id': [46276410, 46276410, 955632, 955632, 1337620, 40168303, 1367268,
                                1314924, 1378382,
                                1314924, 1378382],
            'drug_exposure_start_datetime': [
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
            'person': get_person(),
            'condition_occurrence': get_condition_occurrence(),
            'procedure_occurrence': procedure,
            'drug_exposure': drug_exposure,
            'visit_occurrence': get_visit_occurrence(),
            'observation_period': get_observation_period(),
            'measurement': measurement
        }
        ingredient_list = [46276410, 45775965, 955632, 1337620, 40168303,
                           1367268, 1314924, 1378382]
        drug_temp, cohort_enhanced = line_generation_preprocess(cohort, ingredient_list,
                                                                omop_tables)
        output = LinesOfTherapy(drug_temp, cohort_enhanced, ingredient_list, nb_of_lines=4)()
        expected = """
            {
                "person_id":{"0":1,"1":1,"2":1,"3":2,"4":3,"5":3},
                "line_number":{"0":1,"1":2,"2":3,"3":1,"4":1,"5":2},
                "regimen_name":{"0":"Fluorouracil, pembrolizumab","1":"irinotecan",
                "2":"Fluorouracil, levoleucovorin","3":"Paclitaxel, gemcitabine",
                "4":"gemcitabine","5":"Paclitaxel"},
                "start_date":{"0":1483228800000,"1":1507593600000,"2":1512864000000,
                "3":1514764800000,"4":1546300800000,"5":1561852800000},
                "end_date":{"0":1507507200000,"1":1512777600000,"2":1512864000000,
                "3":1519776000000,"4":1561766400000,"5":1561852800000}
            }
        """
        expected = pd.read_json(expected)
        expected['start_date'] = pd.to_datetime(expected['start_date'], unit='ms')
        expected['end_date'] = pd.to_datetime(expected['end_date'], unit='ms')
        pd.testing.assert_frame_equal(output, expected)

    def test_line_agg_by_patient(self):
        lot = pd.DataFrame({
            'person_id': [1, 1],
            'start_date': [
                pd.to_datetime('2017-11-10'),
                pd.to_datetime('2017-12-10')
            ],
            'line_number': [1, 2],
            'end_date': [
                pd.to_datetime('2017-12-10'),
                pd.to_datetime('2018-04-10')
            ],
            'regimen_name': ['pembrolizumab', 'cabozantinib']
        })
        output = agg_lot_by_patient(lot)
        expected = pd.DataFrame({
            'person_id': [1],
            'start_date 1L': [pd.to_datetime('2017-11-10')],
            'end_date 1L': [pd.to_datetime('2017-12-10')],
            'regimen 1L': ['pembrolizumab'],
            'start_date 2L': [pd.to_datetime('2017-12-10')],
            'end_date 2L': [pd.to_datetime('2018-04-10')],
            'regimen 2L': ['cabozantinib'],
            'start_date 3L': [pd.NaT],
            'end_date 3L': [pd.NaT],
            'regimen 3L': ['no 3L']
        })
        pd.testing.assert_frame_equal(output, expected)
