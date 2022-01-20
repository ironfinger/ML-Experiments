#%%
from statistics import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
# %%

df = pd.read_csv('TWOU.csv')
df.head()


# %%

df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y%m%d")

# %%
df.head()
# %%

x = np.array(df["Date"])
y = np.array(df["Open"])


# %%
print(x)
print(y)
# %%

plt.plot(x, y)
plt.show()
# %%

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=8, include_bias=False)
poly_f = poly.fit_transform(x.reshape(-1,1))

# %%
from sklearn.linear_model import LinearRegression
poly_reg_model = LinearRegression()
poly_reg_model.fit(poly_f, y)
y_predicted = poly_reg_model.predict(poly_f)
# %%

plt.figure(figsize=(10, 6))
plt.title("5th degree polynomial regression", size=16)
plt.plot(x, y)
plt.plot(x, y_predicted, c="red")
plt.show()

# %%
