import pandas as pd
df = pd.read_csv('output.csv')
cycles = df["cycles"][1:]

print(min(cycles)/100)
print(max(cycles)/100)