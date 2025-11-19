from evidently import Report, DataDefinition, Dataset
from evidently.presets import DataDriftPreset
import pandas as pd
import json

df1 = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
df2 = pd.DataFrame({'a': [7,8,9], 'b': [10,11,12]})
dd = DataDefinition(numerical_columns=['a', 'b'])
ds1 = Dataset.from_pandas(df1, data_definition=dd)
ds2 = Dataset.from_pandas(df2, data_definition=dd)
r = Report([DataDriftPreset()])
my_eval = r.run(ds1, ds2)
report_dict = json.loads(my_eval.json())

for i, m in enumerate(report_dict['metrics']):
    print(f"Metric {i}: {m['metric_name']}")
    if isinstance(m['value'], dict):
        print(f"  Value keys: {list(m['value'].keys())}")
        if 'count' in m['value'] and 'share' in m['value']:
            print(f"  Drifted columns count: {m['value']['count']}")
            print(f"  Drift share: {m['value']['share']}")
    else:
        print(f"  Value: {m['value']}")
    print()
