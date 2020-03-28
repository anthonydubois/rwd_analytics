import pandas as pd


class CleaningLabResults():
    def __init__(self, df):
        self.df = df
        
    def test_distance_from_range(self, row):
        if row['distance_from_range'] > row['new_distance_from_range']:
            self.i = self.i + 1
            return row['new_value_as_number']
        else:
            return row['value_as_number']
    
    def __call__(self):
        measurement_dfs = []
        for concept in self.df['measurement_concept_id'].unique().tolist():
            df = self.df[self.df['measurement_concept_id']==concept].copy()
            high = df[df['range_high']!=0]['range_high'].median()
            low = df[df['range_low']!=0]['range_low'].median()
            range_average = (high+low)/2
            df['range_high'] = high
            df['range_low'] = low
            
            self.i = 1
            while self.i != 0:
                self.i = 0
                df['distance_from_range'] = abs(df['value_as_number']-range_average)
                df['new_value_as_number'] = df['value_as_number']*10
                df['new_distance_from_range'] = abs(df['new_value_as_number']-range_average)
                df['value_as_number'] = df.apply(self.test_distance_from_range, axis=1)

            self.i = 1
            while self.i != 0:
                self.i = 0
                df['distance_from_range'] = abs(df['value_as_number']-range_average)
                df['new_value_as_number'] = df['value_as_number']/10
                df['new_distance_from_range'] = abs(df['new_value_as_number']-range_average)
                df['value_as_number'] = df.apply(self.test_distance_from_range, axis=1)

            measurement_dfs.append(df)
        measurement = pd.concat(measurement_dfs)
        measurement = measurement.round({'value_as_number': 1, 'range_high': 1, 'range_low': 1})
        del measurement['distance_from_range']
        del measurement['new_value_as_number']
        del measurement['new_distance_from_range']
        return measurement