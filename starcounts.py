import numpy as np
import matplotlib.pyplot as plt

import getdata as data

def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.0, 1.005*height,
                '%d' % int(height), ha='center', va='bottom', fontsize=14)

fulldata = data.get_data("yelpdata.csv")
xTrain, xTest, yTrain, yTest = data.split_data(fulldata)

allY = np.append(yTrain, yTest)
unique, counts = np.unique(allY, return_counts=True)
full = dict(zip(unique, counts))

print (unique)
print (counts)
print (full)


y_pos = np.arange(len(unique))

fig, ax = plt.subplots()
rects = plt.bar(y_pos, counts, align='center', alpha=0.5, color='g')
autolabel(rects, ax)

plt.xticks(y_pos, unique)
plt.xlabel('Number of Stars Given', fontsize=16)
plt.ylabel('Number of Reviews', fontsize=16)
plt.title('Number of Reviews For Each Amount of Stars Given', fontsize=24)

plt.show()
