import pandas as pd


class EraCalculation():
    def __init__(self, cohort, table, concept_ids=None):
        """
        This function calculates era from first to last records.

        - table is a dask dataframe: condition_occurrence, drug_exposure
        - concept_ids is a list - if not provided, it is done on all concept_ids
        """
        self.table = table
        self.cohort = cohort
        self.subject = self.cohort.person_id.unique().tolist()
        self.table = self.table.loc[self.table.index.isin(self.subject)]
        self.concept_ids = concept_ids

    def __call__(self):
        if 'condition_concept_id' in self.table.columns:
            self.table  = self.table.rename(columns = {
                'condition_concept_id':'concept_id',
                'condition_start_datetime':'start_date'
            })
        elif 'drug_concept_id' in self.table.columns:
            self.table  = self.table.rename(columns = {
                'drug_concept_id':'concept_id',
                'drug_exposure_start_datetime':'start_date'
            })

        self.table = self.table.reset_index()
        t = self.table[['person_id', 'concept_id', 'start_date']]

        if self.concept_ids is not None:
            t = t[t['concept_id'].isin(self.concept_ids)]
        
        t = t.compute()
        t['previous_start_date'] = t['start_date'].shift()
        t['gap_time'] = (t['start_date'] - t['previous_start_date']).dt.days
        t['previous_start_date'] = t['start_date'].shift()
        t['gap_time'] = (t['start_date'] - t['previous_start_date']).dt.days
        t['gap'] = t['gap_time'].apply(lambda x:1 if x > 40 else 0)
        era = t.groupby(['person_id', 'concept_id']).agg(
            start_date_min=pd.NamedAgg(column='start_date', aggfunc=min),
            start_date_max=pd.NamedAgg(column='start_date', aggfunc=max),
            count_exposure=pd.NamedAgg(column='start_date', aggfunc='count'),
            gaps_count=pd.NamedAgg(column='gap', aggfunc=sum)
        )
        era['era_duration'] = (era['start_date_max'] - era['start_date_min']).dt.days
        era = era.reset_index()
        return era


def era_statistics(era):
    era = era.groupby('concept_id').agg({
        'count':['min', 'max', 'mean', 'std'],
        'era_duration':['min', 'max', 'mean', 'std']
    })
    era = round(era, 2)
    return era