import numpy as np


# using numpy ---------------------------------------------------------
x = np.array([1,3,5,7,8,9, 10, 15])
y = np.array([10, 20, 30, 40, 50, 60, 70, 80])

print(np.corrcoef(x, y))


# using pandas

import pandas as pd
from sklearn.datasets import load_diabetes
import seaborn as sns
import matplotlib.pyplot as plt

df = load_diabetes(as_frame=True)
df = df.frame
df.head()
corr = df.corr()
print(corr)

print('subset correlation')
print(np.corrcoef(df['age'],df['sex']))
plt.figure()
sns.heatmap(corr)
plt.show()
