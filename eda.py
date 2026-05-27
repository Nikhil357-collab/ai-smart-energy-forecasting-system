import pandas as pd

# Load dataset
data = pd.read_csv(
    "AI_Powerd_Energy\\data\\archive (6)\\energy.csv",
    parse_dates=['Datetime']
)

# Set datetime index
data.set_index('Datetime', inplace=True)
# Rename energy column automatically
data.rename(
    columns={data.columns[0]: 'Energy'},
    inplace=True
)
# Rename column
data.columns = ['Energy']

# Resample hourly
data = data.resample('H').mean()

# Fill missing values
data = data.ffill()

print(data.head())
print(data.columns)
print(data.info())