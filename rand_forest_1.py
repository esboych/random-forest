# following up ideas from 
# https://medium.com/@maryamuzakariya/project-predict-stock-prices-using-random-forest-regression-model-in-python-fbe4edf01664

import pandas as pd 
import numpy as np 
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV

### 1. read csv file with OHLCV data
df = pd.read_csv('binance_ohlcv_1yrs.csv')
df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
#print(df.to_markdown())

### 2. Visulaise
# df.set_index("Date", inplace=True)
# df['close'].plot()
# plt.ylabel("Adjusted Close Prices")
# plt.show()

### 3. Calc x and y
df.set_index("open_time", inplace=True)
df.dropna(inplace=True)
#print(df.to_markdown())

x = df.iloc[:, 0:6].values
y = df.iloc[:, 3].values
print("x:", x, "\ny:", y)
print("x.shape, y.shape:", x.shape, y.shape)

### 4. Divide into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.26,  random_state=0)
print("x_train, x_test, y_train, y_test:\n", x_train, x_test, y_train, y_test)

### 5. Scaling the features
scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)

### 6. Apply the model
print("Start training..")
model = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_split=2, min_samples_leaf=1, max_depth=10, bootstrap=True)
model.fit(x_train, y_train)
predict = model.predict(x_test)
print("predict:", predict)
print("y_test:", y_test)
print("predict.shape:", predict.shape)

### 7. Calc metrics
print("Mean Absolute Error:", round(metrics.mean_absolute_error(y_test, predict), 4))
print("Mean Squared Error:", round(metrics.mean_squared_error(y_test, predict), 4))
print("Root Mean Squared Error:", round(np.sqrt(metrics.mean_squared_error(y_test, predict)), 4))
print("(R^2) Score:", round(metrics.r2_score(y_test, predict), 4))
print(f'Train Score : {model.score(x_train, y_train) * 100:.2f}% and Test Score : {model.score(x_test, y_test) * 100:.2f}% using Random Tree Regressor.')
errors = abs(predict - y_test)
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.') 