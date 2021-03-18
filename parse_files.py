import pandas as pd
import re

data = pd.read_csv("server_output.csv", index_col=False)

file1 = open('server_output.txt', 'r')

lines = file1.readlines()

i = 0

for line in lines[6:-1]:

    print("row : ", i)

    i += 1

    final_data = []

    data_split = line.split(' ')

    for string in data_split:

        if bool(re.search(r'\d', string)):

            final_data.append(string)

    row = {"Device": 1, "Time": 2.0, "Packet size": float(
        final_data[2]), "Bandwidth": float(final_data[3])}

    data = data.append(row, ignore_index=True)


data.to_csv("server_output.csv", index=False)
