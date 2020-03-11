import pandas as pd
import dask.dataframe as dd

from rwd_analytics.lookups import Descendants, ConceptInfo,  Ingredient


def last_activity_date(cohort, omop_tables):
    subject = cohort.person_id.unique().tolist()
    tables = []
    for table in [omop_tables['drug_exposure'], omop_tables['condition_occurrence']]:
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


def line_generation_preprocess(cohort, ingredient_list, omop_tables):
    subjects = cohort.person_id.unique().tolist()
    descendants = Descendants()
    ingredients = Ingredient()
    drug_temp = omop_tables['drug_exposure'].loc[omop_tables['drug_exposure'].index.isin(subjects)]
    drug_temp = drug_temp[drug_temp['drug_concept_id'].isin(descendants(ingredient_list))]
    drug_temp = drug_temp.compute().reset_index()
    drug_temp = ingredients(drug_temp)
    drug_temp = dd.from_pandas(drug_temp, npartitions=1).set_index('person_id')
    last_activity = last_activity_date(cohort, omop_tables)
    cohort_enhanced = pd.merge(cohort, last_activity, how='left', on='person_id')
    cohort_enhanced = dd.from_pandas(cohort_enhanced, npartitions=1).set_index('person_id')
    return drug_temp, cohort_enhanced


def false_after_lot(x):
    x = x.sort_values(by=['time_from_start'])
    first_false = x['is_in_line'].idxmin()
    if not x['is_in_line'][first_false]:
        x.loc[first_false:, 'is_in_line'] = False
    return x


def is_in_line(drug_code, regimen_codes):
    """
    fluorouracil = 955632
    capecitabine = 1337620
    leucovorin = 1388796
    levoleucovorin = 40168303
    """
    substitutes = {
        955632:[955632, 1337620],
        1337620:[955632, 1337620],
        1388796:[1388796, 40168303],
        40168303:[1388796, 40168303]
    }
    if drug_code not in substitutes:
        substitutes[drug_code] = [drug_code]

    for substitute in substitutes[drug_code]:
        if substitute in regimen_codes:
            return True

    # Addition of leucovorin or levoleucovorin does not advance the lot
    return drug_code in [1388796, 40168303]


class LinesOfTherapy():
    def __init__(self, drug_temp, cohort, offset=14, nb_of_lines = 3):
        self.drug_temp = drug_temp
        self.index_date = cohort
        self.lines = self.__get_drugs(self.drug_temp, self.index_date, offset)
        self.lines = self.lines.persist()
        self.lines['line_number'] = 0
        self.nb_of_lines = nb_of_lines

    def __get_drugs(self, df, index, offset):
        df = dd.merge(df, index, how='left', on='person_id')
        df = df[(df['drug_exposure_start_datetime'] - df['cohort_start_date']).dt.days >= -offset]
        return df[['drug_concept_id', 'drug_exposure_start_datetime']]

    def __get_lines(self, df, line_number):
        def add_paclitaxel_gemcitabine(x):
            """
            Adding Paclitaxel (1378382) or Gemcitabine (1314924)
            does not change the line of therapy
            """
            if (1378382 in x[0]) or (1314924 in x[0]):
                g = list(set(x[1]).symmetric_difference(set(x[0])))
                if 1378382 in g:
                    return np.append(x[0], 1378382)
                elif 1314924 in g:
                    return np.append(x[0], 1314924)
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
        
        while line_number != self.nb_of_lines+1:
            df = lines.map_partitions(self.__get_lines, line_number)
            temp = df[df['is_in_line'] == True][['person_id', 'start_date', 'regimen_codes', 'line_number']]
            lines = df[df['is_in_line'] == False][['person_id', 'drug_concept_id', 'drug_exposure_start_datetime']]
            dfs.append(temp)
            line_number = line_number + 1

        lines_f = dd.concat(dfs)
        lines_f['regimen_codes'] = lines_f['regimen_codes'].astype(str)
        return lines_f.drop_duplicates()


def listToString(s):  
    str1 = ", " 
    return (str1.join(s))


class LineName():
    def __init__(self, lot_f, concept_infos):
        lot_f["regimen_codes"] = lot_f.regimen_codes.str.replace(" ", ',')
        lot_f["regimen_codes"] = lot_f.regimen_codes.str.replace("[", "")
        lot_f["regimen_codes"] = lot_f.regimen_codes.str.replace("]", "")

        for index, row in concept_infos.iterrows():
            lot_f["regimen_codes"] = lot_f.regimen_codes.str.replace(str(row['concept_id']), row['concept_name'])
        self.lot = lot_f

    def __call__(self):
        self.lot['regimen_codes_sorted'] = self.lot['regimen_codes'].str.split(',')
        self.lot = self.lot.reset_index(drop=True)
        for index, row in self.lot.iterrows():
            row['regimen_codes_sorted'].sort()
            regimen_sorted = listToString(row['regimen_codes_sorted'])
            self.lot.loc[index, 'regimen_codes_sorted'] = regimen_sorted

        del self.lot['regimen_codes']
        self.lot = self.lot.sort_values(by=['person_id', 'line_number'])
        self.lot['regimen_codes_sorted'] = self.lot['regimen_codes_sorted'].map(
            lambda x:', '.join([l.strip() for l in x.split(',') if l.strip() != '']))

        return self.lot.rename(columns={'regimen_codes_sorted':'regimen_name'})
    