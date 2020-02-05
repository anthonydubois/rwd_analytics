import pandas as pd

class CohortBuilder():
    def __init__(self, cohort_criteria, drug_exposure, condition_occurrence, concept_ancestor):
        self.conditions = condition_occurrence[
            ['person_id', 'condition_concept_id', 'condition_start_date']]
        self.conditions = self.conditions.rename(columns={
            'condition_concept_id':'concept_id',
            'condition_start_date':'cohort_start_date'
        })

        self.drugs = drug_exposure[
            ['person_id', 'drug_concept_id', 'drug_exposure_start_date']]
        self.drugs = self.drugs.rename(columns={
            'drug_concept_id':'concept_id',
            'drug_exposure_start_date':'cohort_start_date'
        })

        self.concept_ancestor = concept_ancestor
        self.cohort_criteria = cohort_criteria
        self.cohort = pd.DataFrame(columns=['person_id', 'cohort_start_date'])

    def __get_descendant_id(self, concept_ids):
        """
        concept_ids is a list of concept_id
        """
        df = self.concept_ancestor[self.concept_ancestor['ancestor_concept_id'].isin(concept_ids)]
        return df.descendant_concept_id.unique().compute().tolist()

    def __add_criteria(self, tmp, cohort, concept_ids, excluded, descendant, mapped, attributes):
        if descendant == 1:
            concept_ids = self.__get_descendant_id(concept_ids)

        tmp = tmp[tmp['concept_id'].isin(concept_ids)].compute()
        
        if len(attributes) != 0:
            for attribute in attributes:
                if attribute['type'] == 'occurrence':
                    t = pd.DataFrame(tmp.groupby('person_id').cohort_start_date.nunique()).reset_index()
                    t.columns = ['person_id', 'count']
                    t = t[t['count'] >= attribute['feature']]
                    tmp = tmp[tmp['person_id'].isin(t.person_id.tolist())]

        if excluded == 0:
            cohort = cohort[cohort['person_id'].isin(tmp.person_id.tolist())]
            tmp = pd.DataFrame(tmp.groupby('person_id').cohort_start_date.min())
            tmp = tmp.reset_index()
            cohort = pd.concat([cohort, tmp])
            cohort = cohort.groupby('person_id').cohort_start_date.max()
            cohort = pd.DataFrame(cohort).reset_index()
        else:
            cohort = cohort[~cohort['person_id'].isin(tmp.person_id.unique().tolist())]

        return cohort

    def __call__(self):
        for criteria in self.cohort_criteria['criteria']:
            concept_type = criteria['concept_type']
            concept_id = criteria['concept_id']
            excluded = criteria['excluded']
            descendant = criteria['descendant']
            mapped = criteria['mapped']
            attributes = criteria['attributes']

            if concept_type == 'condition':
                tmp = self.conditions
            
            if concept_type == 'drug':
                tmp = self.drugs

            self.cohort = self.__add_criteria(tmp, self.cohort, concept_id,
                                              excluded, descendant, mapped, attributes)
        
        return self.cohort
