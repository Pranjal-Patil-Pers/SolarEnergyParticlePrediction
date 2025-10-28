import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

df = pd.read_csv('../data/raw/data/1986-02-07_01-00.csv', delimiter=',')
plt.plot(df)
plt.show()
