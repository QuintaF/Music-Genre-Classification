
import pandas as pd
import numpy as np

a = np.array([[4,5,6,7,8],[2,3,4,6,7]])
s = np.array2string(a, separator=';').replace('\n','')
a = np.array([[s, s], [s, s]])
df = pd.DataFrame(a)

df.to_csv("maronn.csv")