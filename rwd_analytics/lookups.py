import pandas as pd


class Descendants():
    def __init__(self):
        #self.concept_ancestor = pd.read_csv('')
        self.concept_ancestor = pd.DataFrame()

    def __call__(self, concept_ids):
        """
        concept_ids is a list of concept_id
        """
        df = self.concept_ancestor[self.concept_ancestor['ancestor_concept_id'].isin(concept_ids)]
        return df.descendant_concept_id.unique().tolist()


class ComorbidConditions():
    def __init__(self):
        self.comorbidities = pd.read_csv('resources/comorbid_conditions/comorbidities_magic.csv')

    def __call__(self):
        return self.comorbidities