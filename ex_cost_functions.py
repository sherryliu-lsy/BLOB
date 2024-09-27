from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

cgpa = [6.89, 5.12, 7.82, 7.42, 6.94, 7.89, 6.73, 6.75, 6.09]
package = [3.26, 1.98, 3.25, 3.67, 3.57, 2.99, 2.6, 2.48, 2.31]
df = pd.DataFrame({'cgpa' : cgpa, 'package' : package})
y = df['package']
X = df.drop('package', axis = 1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print(y_pred)

print("MAE",mean_absolute_error(y_test,y_pred))
print("MSE",(mean_squared_error(y_test,y_pred)))
print("RMSE",np.sqrt(mean_squared_error(y_test,y_pred)))
print("LRMSE",np.sqrt(mean_squared_error(y_test,y_pred)))
print("RMSLE", np.sqrt(mean_squared_error(np.log1p(y_test), np.log1p(y_pred))))