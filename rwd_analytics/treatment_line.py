import pandas as pd


def last_activity_date(cohort, drug_exposure, condition_occurrence):
    subject = cohort.person_id.unique().tolist()
    tables = []
    for table in [drug_exposure, condition_occurrence]:
        if 'condition_concept_id' in table.columns:
            table  = table.rename(columns = {
                'condition_start_datetime':'start_date'
            })
        elif 'drug_concept_id' in table.columns:
            table  = table.rename(columns = {
                'drug_exposure_start_datetime':'start_date'
            })
            
        table = table.loc[table.index.isin(subject)]
        table = table[['start_date']]
        table = table.compute().reset_index()
        table = table.groupby('person_id').start_date.max().to_frame()
        tables.append(table)
    
    max_dates = pd.concat(tables)
    max_dates = max_dates.groupby('person_id').start_date.max().to_frame()
    max_dates.columns = ['last_activity_date']
    return max_dates.reset_index()


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


class LinesOfTherapy():
    def __init__(self, drug_exposure, eligible_therapies, index_date):
        self.drug_exposure = drug_exposure
        self.eligible_therapies = eligible_therapies
        self.index_date = index_date
        self.lines = self.__get_eligible_therapies(self.drug_exposure, self.eligible_therapies)
        self.lines = self.__get_drugs(self.lines, self.index_date, offset=14)
        self.lines = self.lines.persist()
        self.lines['line_number'] = 0

    def __get_eligible_therapies(self, df, eligible_therapies):
        return df[df['drug_concept_id'].isin(eligible_therapies)]

    def __get_drugs(self, df, index, offset=14):
        df = dd.merge(df, index, how='left', on='person_id')
        df = df[(df['drug_exposure_start_datetime'] - df['cohort_start_date']).dt.days >= -offset]
        return df[['drug_concept_id', 'drug_exposure_start_datetime']]

    def __get_lines(self, df, line_number):
        def add_paclitaxel_gemcitabine(x):
            if (paclitaxel in x[0]) or (gemcitabine in x[0]):
                g = list(set(x[1]).symmetric_difference(set(x[0])))
                if paclitaxel in g:
                    return np.append(x[0], paclitaxel)
                elif gemcitabine in g:
                    return np.append(x[0], gemcitabine)
            return x[0]
        
        df = df.reset_index()
        start_line = df.groupby(['person_id'])['drug_exposure_start_datetime'].min().to_frame('start_date')
        df = df.merge(start_line, how='left', on='person_id')
        df['time_from_start'] = (df['drug_exposure_start_datetime'] - df['start_date']).dt.days
        regimen_codes = df[df['time_from_start'] <= 28].groupby('person_id').drug_concept_id.unique().to_frame('regimen_codes_28')
        df = df.merge(regimen_codes, how='left', on='person_id')
        regimen_codes = df[df['time_from_start'] <= 90].groupby('person_id').drug_concept_id.unique().to_frame('regimen_codes_90')
        df = df.merge(regimen_codes, how='left', on='person_id')    
        df['regimen_codes'] = df['regimen_codes_28']
        df['tmp'] = list(zip(df['regimen_codes_28'], df['regimen_codes_90']))
        df['regimen_codes'] = df['tmp'].map(add_paclitaxel_gemcitabine)
        df = df.sort_values(by=['person_id', 'time_from_start'])
        df['tmp'] = list(zip(df['drug_concept_id'], df['regimen_codes']))
        df['is_in_line'] = df['tmp'].map(lambda x: is_in_line(x[0], x[1]))
        del df['tmp']
        df = df.groupby('person_id').apply(false_after_lot).reset_index(drop=True)
        df['line_number'] = line_number
        df['line_number'] = df['line_number'].astype(int)
        return df
    
    def __call__(self):
        line_number = 1
        dfs = []
        tmp = []
        lines = self.lines
        
        while line_number != 4:
            df = lines.map_partitions(self.__get_lines, line_number)
            temp = df[df['is_in_line'] == True][['person_id', 'start_date', 'regimen_codes', 'line_number']]
            lines = df[df['is_in_line'] == False][['person_id', 'drug_concept_id', 'drug_exposure_start_datetime']]
            dfs.append(temp)
            line_number = line_number + 1

        lines_f = dd.concat(dfs)
        lines_f['regimen_codes'] = lines_f['regimen_codes'].astype(str)
        return lines_f.drop_duplicates()