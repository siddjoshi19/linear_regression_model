# linear_regression_model
Basics of Linear Regression model. We will predict the sales using Linear Regression model for a small dataset, and visualizing it using plot.

## 1. Import necessary libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
```
## 2. Creating Small database of Sales for 10 Years
```python
data={ 'years': list(range(2001,2011)),
       'sales':[1000, 1200, 1300, 1500, 1700, 2000, 2200, 2500, 2800, 3000]
     }

df=pd.DataFrame(data)
```

## 3. Split data in train and test datasets
```python
x=df[['years']]
y=df[['sales']]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
print(x_train, x_test, y_train, y_test)
```

## 4. Create and Train the Model
```python
lr=LinearRegression() #creates a model, and then from below; we trained our model using train dataset)
lr.fit(x_train, y_train)
```

### NOTE: we can check the value of m and b for our linear regression equation : y=mx+b, where m is slope and b is intercept
```python
print(" Slope(m): ", lr.coef_)
print("Intercept(b): ", lr.intercept_)
```

## Now test the model using test data
```python
y_lr_train_predic=lr.predict(x_train)
y_predic= lr.predict(x_test)
print("y_predic value: ", y_predic)
```

## 5. Evaluate the model using MSE and R2
```python

mse= mean_squared_error(y_test, y_predic)
r2=r2_score(y_test, y_predic)

print("Mean squarred error: ", mse)
print("R2_score: ", r2)
```
### Note: if you are testing by multiple models, then whichever model has lower MSE value and R2_Score closest to 1, is considered as accurate Model.

## 6. Visualize the result
```pyhton
plt.scatter(x,y, label='Actual Sales')
plt.plot(x, lr.predict(x), color='red', linewidth=2, label='Regression Line')
plt.xlabel('Years')
plt.ylabel('Sales')
plt.legend()
plt.title('Linear Regresssion: Sales Prediction')
plt.show()
```

## 7. Make future sales prediction from the model
```python
future_year=[[2015],[2020]]
future_sales=lr.predict(future_year)
```
### Plot 
```python
plt.scatter(x, y, color='pink', label='Original Sales Data')
plt.scatter(x_train, y_lr_train_predic, color = 'green', label = 'Training Results', marker = 'o')
plt.scatter(x_test, y_predic, color= 'red')
plt.scatter([2015,2020], future_sales, color='Orange', marker ='+')
years_extended=df['years'].tolist()+[2015,2020]
sales_extened=df['sales'].tolist()+list(future_sales)
import numpy as np

# Ensure Years_extended is reshaped to 2D format for predictions
years_extended = np.array(years_extended).reshape(-1, 1) #from chatGPT 
plt.plot(years_extended, lr.predict(years_extended), linestyle='--', color='Black', linewidth='1')

```


