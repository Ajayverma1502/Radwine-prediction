# Radwine-prediction-
This is machine learning project
<br>Author praveen prajapat<br>

import pandas as pd
import matplotlib.pyplot as plt


from sklearn import linear_model

df=pd.read_csv('redwine.csv')

print(df)


reg=linear_model.LinearRegression()

reg.fit(df[['fixed acidity','alcohol','free sulfur dioxide']],df.quality)

print(reg.predict([[8.9,9.3,22]]))

