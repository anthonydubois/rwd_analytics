import dask.dataframe as dd
import pandas as pd
import s3fs

from datetime import datetime
fs = s3fs.S3FileSystem()

def load_omop_table(dataset_path):
    omop_table_field = [
        {
            'table':'person',
            'isRequired':['year_of_birth', 'gender_concept_id'],
            'isDatetime':[],
            'isConceptId':['gender_concept_id']
        },
        {
            'table':'condition_occurrence',
            'isRequired':['condition_concept_id', 'condition_start_datetime',
                            'condition_source_value', 'condition_source_concept_id'],
            'isDatetime':['condition_start_datetime'],
            'isConceptId':['condition_concept_id', 'condition_source_concept_id']
        },
        {
            'table':'procedure_occurrence',
            'isRequired':['procedure_concept_id', 'procedure_datetime',
                            'procedure_source_value', 'procedure_source_concept_id'],
            'isDatetime':['procedure_datetime'],
            'isConceptId':['procedure_concept_id', 'procedure_source_concept_id']
        },
        {
            'table':'drug_exposure',
            'isRequired':['drug_concept_id', 'drug_exposure_start_datetime',
                            'drug_source_value', 'drug_source_concept_id'],
            'isDatetime':['drug_exposure_start_datetime'],
            'isConceptId':['drug_concept_id', 'drug_source_concept_id']
        },
        {
            'table':'visit_occurrence',
            'isRequired':['visit_start_date'],
            'isDatetime':['visit_start_date'],
            'isConceptId':[],
            'rename':[{'old':'visit_start_date',
                      'nex':'visit_start_datetime'}]
        },
        {
            'table':'observation_period',
            'isRequired':['observation_period_start_date', 'observation_period_end_date'],
            'isDatetime':['observation_period_start_date', 'observation_period_end_date'],
            'isConceptId':[]
        },
        {
            'table':'measurement',
            'isRequired':['measurement_concept_id', 'measurement_datetime', 'unit_source_value',
                          'value_source_value', 'value_as_number', 'range_low', 'range_high'],
            'isDatetime':['measurement_datetime'],
            'isConceptId':['measurement_concept_id']
        },
        {
            'table':'observation',
            'isRequired':['observation_concept_id', 'observation_datetime'],
            'isDatetime':['observation_datetime'],
            'isConceptId':['observation_concept_id']
        }
    ]
    omop_files_tmp = []
    for omop_table in omop_table_field:
        try:
            tmp = dd.read_parquet(dataset_path+omop_table['table'], engine='pyarrow')
            tmp = tmp[omop_table['isRequired']]
            for date in omop_table['isDatetime']:
                tmp[date] = dd.to_datetime(tmp[date])

            for concept in omop_table['isConceptId']:
                tmp[concept] = tmp[concept].astype(int)
        except:
            tmp = pd.DataFrame(columns=omop_table['isRequired'])
            tmp = dd.from_pandas(tmp, npartitions=1)
            print('No '+omop_table['table']+' table available')

        omop_files_tmp.append(tmp)

    omop_tables = {
        'person':omop_files_tmp[0],
        'condition_occurrence':omop_files_tmp[1],
        'procedure_occurrence':omop_files_tmp[2],
        'drug_exposure':omop_files_tmp[3],
        'visit_occurrence':omop_files_tmp[4],
        'observation_period':omop_files_tmp[5],
        'measurement':omop_files_tmp[6],
        'observation':omop_files_tmp[7]
    }
    print ('***********  Data successfully loaded  ***********')
    return omop_tables


def data_extractor(cohort, data_path, output_path=None, output_format='csv'):
    """
    This function extracts all records of all patients in cohort.
    cohort: output of CohortBuilder()
    input_path: raw data or omop in parquet format 
        indexed on patient ID (ENROLID, PATIENT_ID, PERSON_ID, etc.)
    output_path: where the files are being saved
    output_format: 'csv', 'parquet'
    """
    subjects = cohort.patient_id.unique().tolist()
    content = [f'{f_path}' for f_path in fs.ls(data_path)]
    tables = [l.split('/')[-1].lower() for l in content]

    if ('condition_occurrence' in tables) or ('inpatient_services' in tables):
        tables = tables
    elif 'rx_fact' in tables:
        # TODO: should take into account reference tables
        tables = ['rx_fact', 'dx_fact']
    else:
        return 'Format to extract not detected'

    for table in tables:
        print(table)
        try:
            df = dd.read_parquet(data_path + table, engine='pyarrow')
        except:
            df = dd.read_parquet(data_path + table.upper(), engine='pyarrow')
            
        df = df.loc[df.index.isin(subjects)]
        df = df.compute()
        if output_path is not None:
            if output_format == 'csv':
                df.to_csv(output_path+table+'.csv')
            if output_format == 'parquet':
                df.to_parquet(output_path+table)