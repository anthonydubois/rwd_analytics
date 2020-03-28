#!pip install lifelines --user
from lifelines.utils import datetimes_to_durations
from lifelines import KaplanMeierFitter
from lifelines.plotting import plot_lifetimes

from matplotlib import pyplot as plt
%matplotlib inline
%pylab inline


def plot_kp_time_to_next_treatment(lot, line, metric, cohort_enhanced, censoring_date=None,
                                   displayed_regimen=None, timeline='M'):
    """
    Inputs:
    - metric: 'time_to_next_treatment', 'time_to_last_activity'
    """
    kmf = KaplanMeierFitter()
    figsize(10, 8)
    ax = plt.subplot(111)
    if timeline == 'M':
        x_label = 'Duration in months'
    elif timeline == 'D':
        x_label = 'Duration in days'
    elif timeline == 'Y':
        x_label = 'Duration in years'
    
    if metric == 'time_to_last_activity':
        lot = lot.merge(cohort_enhanced, how='left', on='person_id')
        title = "Time to last activity for different regimens"
        end_date = 'last_activity_date'
        
    elif metric == 'time_to_next_treatment':
        title = "Time to next treatment for different regimens"
        end_date = 'end_date'
    
    df = lot[lot['line_number']==line]
    groups = df['regimen_name']
    
    if displayed_regimen is None:
        labels = groups.unique().tolist()
    else:
        labels = displayed_regimen

    for a, label in enumerate(labels):
        i = (groups == label)
        start_dates = df[i]['start_date']
        end_dates = df[i][end_date]
        T, E = datetimes_to_durations(start_dates, end_dates, fill_date = censoring_date, freq=timeline)
        kmf.fit(T, event_observed=E, label=label)
        kmf.plot(ax=ax)
        
    plt.ylim(0, 1)
    plt.xlim(0, 20)
    plt.xlabel(x_label)
    plt.title(title)

censoring_date='2019-01-31'
line = 1
metric = 'time_to_next_treatment'
displayed_regimen=['cabozantinib', 'sunitinib']
plot_kp_time_to_next_treatment(lot, line, metric, cohort_enhanced, censoring_date,
                               displayed_regimen, timeline)
            
            
def describe_lot(lot, cohort_enhanced, line, censoring_date):
    lot = lot.merge(cohort_enhanced, how='left', on='person_id')
    lot['time_to_last_activity'] = (lot['last_activity_date'] - lot['start_date']).dt.days
    lot['time_to_next_treatment'] = (lot['end_date'] - lot['start_date']).dt.days
    lot['event'] = lot['end_date'].apply(lambda x:0 if x > pd.to_datetime(censoring_date, format='%Y-%m-%d') else 1)
    lot = lot[lot['event']==1]
    lot = lot[lot['line_number']==line]
    lot = lot.groupby('regimen_name').agg({
        'person_id':['count'],
        'time_to_next_treatment':['median'],
        'time_to_last_activity':['median']
    })
    lot = lot[lot[('person_id', 'count')]>=50]
    return lot.sort_values(by=[('time_to_next_treatment', 'median')], ascending=False)
