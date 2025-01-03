# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:03:53 2024

@author: Muberra
"""

#import library
import pandas as pd
import matplotlib.pyplot as plt

#import data
df = pd.read_csv("linear_regression_dataset.csv", sep = ";")

#plot data
plt.scatter(df.deneyim, df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()


#%% linear regression

#sklearn library
from sklearn.linear_model import LinearRegression

#linear regression model
linear_reg = LinearRegression()

x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

y_head = linear_reg.predict(x)

#½½

from sklearn.metrics import r2_score

print("r_score: " , r2_score(y,y_head))