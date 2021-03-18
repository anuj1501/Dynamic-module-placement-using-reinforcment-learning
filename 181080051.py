import numpy as np

import pandas as pd

data = pd.read_csv('dataset.csv')

examples = np.array(data.iloc[:, :-1])

print("\nInstances are:\n", examples)

target = np.array(data.iloc[:, -1])

print("\nTarget Values are: ", target)


def learn(examples, target):

    specific_h = examples[0].copy()

    print("\nSpecific Hypothesis: ", specific_h)

    general_h = [["?" for i in range(len(specific_h))]
                 for i in range(len(specific_h))]

    print("\nGeneral Hypothesis: ", general_h)

    for i, h in enumerate(examples):

        print("\nInstance", i+1, "is ", h)

        if target[i] == "Yes":

            print("Instance is Positive ")

            for x in range(len(specific_h)):

                if h[x] != specific_h[x]:

                    specific_h[x] = '?'

                    general_h[x][x] = '?'

        if target[i] == "No":

            print("Instance is Negative ")

            for x in range(len(specific_h)):

                if h[x] == specific_h[x]:

                    general_h[x][x] = '?'

                else:
                    general_h[x][x] = specific_h[x]

        print("Specific Boundary after ", i+1, "Instance is ", specific_h)

        print("Generic Boundary after ", i+1, "Instance is ", general_h)

        print("\n")

    indices = [i for i, val in enumerate(general_h) if val == [
        '?', '?', '?', '?', '?', '?', '?', '?', '?']]

    for i in indices:

        general_h.remove(['?', '?', '?', '?', '?', '?', '?', '?', '?'])

    return specific_h, general_h


s_final, g_final = learn(examples, target)

print("Final Specific Hypothesis: \n", s_final)

print("Final General Hypothesis: \n", g_final)

version_space = []

count_x = -1

for x in s_final:

    count_x = count_x+1

    if x != '?' and g_final[0][count_x] != x:

        hypo = ['?']*len(s_final)

        hypo[count_x] = x

        count_y = 0

        for y in g_final[0]:

            count_y = count_y + 1

            if y != '?':

                hypo[count_y] = y

                version_space.append(hypo)

print('Final Version Space:\n')

for x in version_space:

    print(x)
