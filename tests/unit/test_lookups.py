import pandas as pd

from rwd_analytics.lookups import Concept


class TestLookups():
    def test_concept_info(self):
        df = pd.DataFrame({
            'person_id': [1, 2, 3],
            'drug_concept_id': [43012292, 43012292, 43012292]
        })
        concept = Concept(usecols=['concept_id', 'concept_name', 'domain_id'])()
        output = df.merge(concept, how='left',
                          left_on='drug_concept_id', right_on='concept_id')
        del output['concept_id']
        print(output)
        expected = pd.DataFrame({
            'person_id': [1, 2, 3],
            'drug_concept_id': [43012292, 43012292, 43012292],
            'concept_name': ['cabozantinib']*3,
            'domain_id': ['Drug']*3
        })
        pd.testing.assert_frame_equal(output, expected)
