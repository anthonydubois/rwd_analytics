import pandas as pd
import dask.dataframe as dd

OMOP_VOC_PATH = 'resources/omop_voc/'


class Descendants():
    def __init__(self):
        self.concept_ancestor = dd.read_csv(OMOP_VOC_PATH+'CONCEPT_ANCESTOR.csv', sep="\t",
                                            usecols=['descendant_concept_id', 'ancestor_concept_id'])

    def __call__(self, concept_ids):
        """
        concept_ids is a list of concept_id
        """
        df = self.concept_ancestor[self.concept_ancestor['ancestor_concept_id'].isin(concept_ids)]
        return df.descendant_concept_id.unique().compute().tolist()


class ComorbidConditions():
    def __init__(self):
        self.comorbidities = pd.read_csv('resources/comorbid_conditions/comorbidities_magic.csv')

    def __call__(self, comorbid=None):
        """
        Return a list of CONCEPT IDs or all comorbidities.

        Possible comorbid conditions to be requested are:
        'Congestive heart failure', 'Cardiac arrhythmias', 'Valvular disease',
        'Pulmonary circulation Disorders', 'Peripheral vascular disorders', 'Hypertension,uncomplicated',
        'Hypertension,complicated', 'Paralysis', 'Other neurological disorders',
        'Chronic pulmonary disease', 'Diabetes,uncomplicated', 'Diabetes,complicated',
        'Hypothyroidism', 'Renal failure', 'Liver disease', 'Peptic ulcer disease excluding bleeding',
        'Lymphoma', 'Metastatic cancer', 'Solid tumor without metastasis',
        'Rheumatoid arthritis/collagen vascular diseases',
        'Coagulopathy', 'Obesity', 'Weight loss', 'Fluid and electrolyte disorders',
        'Blood loss anemia', 'Deficiency anemia', 'Drug abuse', 'Psychoses',
        'Depression', 'Alcohol abuse', 'AIDS/H1V'
        """
        if comorbid is not None:
            self.comorbidities = self.comorbidities[self.comorbidities['COMMORBIDITIES']==comorbid]
            temp = self.comorbidities.iloc[0]['CONCEPT_ID']
            temp = temp.replace('[', '').replace(']', '').split(', ')
            return temp
        return self.comorbidities


class Ingredient():
    def __init__(self):
        concept = dd.read_csv(OMOP_VOC_PATH+'CONCEPT.csv', sep="\t",
                              dtype={
                                  'standard_concept': 'object',
                                  'vocabulary_id': 'object',
                                  'concept_class_id': 'object',
                                  'invalid_reason': 'object'},
                              usecols=['concept_id', 'vocabulary_id', 'standard_concept',
                                       'invalid_reason', 'concept_class_id'])
        concept = concept[(concept['vocabulary_id'] == 'RxNorm')
                        &(concept['standard_concept']=='S')
                        &(concept['invalid_reason'].isnull())
                        &(concept['concept_class_id'] == 'Ingredient')] 
        self.list_concept = concept['concept_id'].unique().compute().tolist()
        self.concept_ancestor = dd.read_csv(OMOP_VOC_PATH+'CONCEPT_ANCESTOR.csv', sep="\t",
                                        usecols=['descendant_concept_id', 'ancestor_concept_id'])

    def __call__(self, df):
        drug_concept_ids = df.drug_concept_id.unique().tolist()
        temp = self.concept_ancestor[self.concept_ancestor['descendant_concept_id'].isin(drug_concept_ids)]
        temp = temp[temp['ancestor_concept_id'].isin(self.list_concept)]
        temp = temp.compute()
        df = pd.merge(df, temp, how='left',
                      left_on='drug_concept_id', right_on='descendant_concept_id')
        del df['drug_concept_id']
        del df['descendant_concept_id']
        df = df.rename(columns={'ancestor_concept_id':'drug_concept_id'})
        return df


class ConceptRelationship():
    def __init__(self):
        self.concept_relationship = dd.read_csv(OMOP_VOC_PATH+'CONCEPT_RELATIONSHIP.csv', sep="\t")

    def get_standard(self, concept_ids):
        """
        Return a list of standard concept IDs
        Parameters:
            - concept_ids is a list of standards or non standard concept IDS
        """
        df = self.concept_relationship[self.concept_relationship['concept_id_1'].isin(concept_ids)]
        df = df[(df['relationship_id'] == 'Maps to') & (df['invalid_reason'].isnull())]
        return df.concept_id_2.unique().compute().tolist()

    def get_non_standard(self, concept_ids):
        """
        Return a list of non-standard concept IDs
        Parameters:
            - concept_ids is a list of standards concept IDS
        """
        df = self.concept_relationship[self.concept_relationship['concept_id_1'].isin(concept_ids)]
        df = df[(df['relationship_id'] == 'Mapped from') & (df['invalid_reason'].isnull())]
        return df.concept_id_2.unique().compute().tolist()

    def __call__(self, concept_ids):
        tmp = self.concept_relationship[self.concept_relationship['concept_id_1'].isin(concept_ids)]
        return tmp.compute()


class Concept():
    def __init__(self):
        self.concept = dd.read_csv(OMOP_VOC_PATH+'CONCEPT.csv', sep="\t",
                                   dtype={
                                       'standard_concept': 'object',
                                       'concept_code': 'object',
                                       'concept_name': 'object',
                                       'invalid_reason': 'object'
                                       })
        self.concept_relationship = dd.read_csv(OMOP_VOC_PATH+'CONCEPT_RELATIONSHIP.csv', sep="\t")

    def search_for_concept_by_name(self, search_value_string, domain_id=None, standard_concept='S'):
        """
        - search_value_string is a string
        """
        search_value_string = search_value_string.lower()
        self.concept['concept_name'] = self.concept['concept_name'].fillna('')
        df = self.concept[self.concept['concept_name'].str.lower().str.contains(search_value_string)]
        if standard_concept is not None:
            df = df[df['standard_concept']==standard_concept]
        if domain_id is not None:
            df = df[df['domain_id']==domain_id]
        return df.compute()

    def get_unique_concept_name(self, concept_id):
        c = self.concept[self.concept['concept_id']==concept_id].reset_index().compute()
        return c.at[0, 'concept_name']
    
    def get_info(self, df, columns):
        """
        - df is a dataframe
        - columns is a list of columns.
            Valid columns:
            - concept_name
            - domain_id
            - vocabulary_id
            - concept_class_id
            - standard_concept
            - concept_code
            - valid_start_date
            - valid_end_date
            - invalid_reason
        """
        df = df.rename(columns={
            'condition_concept_id':'concept_id',
            'drug_concept_id':'concept_id',
            'measurement_concept_id':'concept_id',
            'observation_concept_id':'concept_id'
        })
        concept_ids = df.concept_id.unique().tolist()
        columns.append('concept_id')
        temp = self.concept[columns]
        temp = temp[temp['concept_id'].isin(concept_ids)]
        temp = temp.compute()
        return df.merge(temp[columns], how='left', on='concept_id')

    def get_concept_id(self, concept_code):
        df = self.concept[self.concept['concept_code'].isin(concept_code)]
        return df.concept_id.unique().compute().tolist()

    def get_standard(self, concept_ids):
        df = self.concept_relationship[self.concept_relationship['concept_id_1'].isin(concept_ids)]
        df = df[(df['relationship_id'] == 'Maps to') & (df['invalid_reason'].isnull())]
        return df.concept_id_2.unique().compute().tolist()

    def __call__(self, concept_ids):
        """
        Returns all information about concept ids

        Parameters: - concept_ids: a list of concept_ids
        """
        df = self.concept[self.concept['concept_id'].isin(concept_ids)]
        return df.compute().reset_index()