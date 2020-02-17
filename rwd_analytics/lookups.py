import pandas as pd
import dask.dataframe as dd

OMOP_VOC_PATH = 'resource/omop_voc/'


class Descendants():
    def __init__(self):
        self.concept_ancestor = dd.read_csv(OMOP_VOC_PATH+'concept_ancestor.csv', sep="\t")

    def __call__(self, concept_ids):
        """
        concept_ids is a list of concept_id
        """
        df = self.concept_ancestor[self.concept_ancestor['ancestor_concept_id'].isin(concept_ids)]
        return df.descendant_concept_id.unique().compute().tolist()


class ComorbidConditions():
    def __init__(self):
        self.comorbidities = pd.read_csv('resources/comorbid_conditions/comorbidities_magic.csv')

    def __call__(self):
        return self.comorbidities


class ConceptInfo():
    def __init__(self):
        self.concept = dd.read_csv(OMOP_VOC_PATH+'concept.csv', sep="\t")
    
    def __call__(self, df, columns):
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
        concept_ids = df.concept_id.unique().tolist()
        columns.append('concept_id')
        temp = self.concept[columns]
        temp = temp[temp['concept_id'].isin(concept_ids)]
        temp = temp.compute()
        return df.merge(temp[columns], how='left', on='concept_id')


class Ingredient():
    def __init__(self, df):
        concept = dd.read_csv(OMOP_VOC_PATH+'concept.csv', sep="\t")
        concept = concept[(concept['vocabulary_id'] == 'RxNorm')
                        &(concept['standard_concept']=='S')
                        &(concept['invalid_reason'].isnull())
                        &(concept['concept_class_id'] == 'Ingredient')] 
        list_concept = concept['concept_id'].unique().compute().tolist()
        drug_concept_ids = df.drug_concept_id.unique().tolist()
        concept_ancestor = dd.read_csv(OMOP_VOC_PATH+'concept_ancestor.csv', sep="\t")
        concept_ancestor = concept_ancestor[concept_ancestor['descendant_concept_id'].isin(drug_concept_ids)]
        self.concept_ancestor = concept_ancestor[concept_ancestor['ancestor_concept_id'].isin(list_concept)]
        self.df = df

    def __call__(self):
        df = pd.merge(self.df, self.concept_ancestor, how='left',
                      left_on='drug_concept_id', right_on='descendant_concept_id')
        del df['drug_concept_id']
        df = df.rename(columns={'ancestor_concept_id':'drug_concept_id'})
        return df


class ConceptRelationship():
    def __init__(self):
        self.concept_relationship = dd.read_csv(OMOP_VOC_PATH+'concept_relationship.csv', sep="\t")

    def __call__(self, concept_ids):
        tmp = self.concept_relationship[self.concept_relationship['concept_id_1'].isin(concept_ids)]
        return tmp.compute()