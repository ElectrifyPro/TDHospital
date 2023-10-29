import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

data_path = './TD_HOSPITAL_TRAIN.csv'
df = pd.read_csv(data_path)
col_to_keep = ['timeknown', 'death', 'reflex','blood', 'bloodchem1', 'bloodchem2', 'temperature', 'heart', 'psych1', 'glucose', 'psych2', 'dose', 'bloodchem3', 'confidence', 'bloodchem4', 'comorbidity', 'breathing', 'age']
df = df[col_to_keep]
df.shape

# a - bool
# b - numeric
a = df['death']
for col in col_to_keep:
  # Set 'b' to the current column
  b = df[col]

  # Convert 'b' to numeric, excluding non-numeric values (e.g., strings)
  b_numeric = pd.to_numeric(b, errors='coerce')
    
  # Exclude rows with NaN or non-numeric values in 'b'
  valid_rows = ~a.isna() & ~b_numeric.isna()
  a_valid = a[valid_rows]
  b_valid = b_numeric[valid_rows]

  # Calculate point-biserial correlation
  point_biserial_corr, _ = stats.pointbiserialr(a_valid, b_valid)
  # Print the result for each column
  print(f'Point-Biserial Correlation between "death" and "{col}": {point_biserial_corr:.4f}')
