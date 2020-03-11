import dask.dataframe as dd
import pandas as pd

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
            'isRequired':['condition_concept_id', 'condition_start_datetime'],
            'isDatetime':['condition_start_datetime'],
            'isConceptId':['condition_concept_id']
        },
        {
            'table':'procedure_occurrence',
            'isRequired':['procedure_concept_id', 'procedure_datetime'],
            'isDatetime':['procedure_datetime'],
            'isConceptId':['procedure_concept_id']
        },
        {
            'table':'drug_exposure',
            'isRequired':['drug_concept_id', 'drug_exposure_start_datetime'],
            'isDatetime':['drug_exposure_start_datetime'],
            'isConceptId':['drug_concept_id']
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
            'isRequired':['measurement_concept_id', 'measurement_datetime'],
            'isDatetime':['measurement_datetime'],
            'isConceptId':['measurement_concept_id']
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
        'measurement':omop_files_tmp[6]
    }
    print ('***********  Data successfully loaded  ***********')
    return omop_tables