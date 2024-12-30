import pandas as pd
import numpy as np
from sklearn.model_selection import KFold,cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler

df = pd.read_excel(r"C:\Users\cdertili\Downloads\housedata.xls", sheet_name='Sheet1')
raw_data = df.replace(';', '', regex=True)
raw_data.info()
raw_data.dtypes
#split the row data into train and test datasets

raw_data['selling price in 1000 dollars'] = raw_data['selling price in 1000 dollars'].astype(float) / 1000
raw_data['house area in 1000 square feet'] = raw_data['house area in 1000 square feet'] / 1000
X = raw_data[['house area in 1000 square feet', '#bedrooms']]
y=raw_data[['selling price in 1000 dollars']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=79, random_state=42)



regressor = LinearRegression() # instantiate the Linear Regression model
regressor.fit(X_train, y_train)

coef_model1=regressor.coef_

y_pred = regressor.predict(X)

# Creating the scatter plot
#plt.scatter(y, predicted_y)
plt.scatter(y, y_pred, c='red', label='Actual Price')
plt.scatter(y,y_pred, c='purple', label='Predicted Price')
plt.xlabel('Actual Sale Price') 
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs. Predicted Sale Prices')
plt.legend()
plt.show()



y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
print("Training RMSE:", train_rmse)
print("Test RMSE:", test_rmse)


test_houses = pd.DataFrame({
    'house area in 1000 square feet': [846, 1324, 1150, 3037, 3984],
    '#bedrooms': [1, 2, 3, 4, 5],
    'Actual selling price in 1000 dollars': [115000, 234500, 198000, 528000, 572500]
})
X_test1 = test_houses[['house area in 1000 square feet', '#bedrooms']]
y_test1 = test_houses[['Actual selling price in 1000 dollars']]
y_pred_test1 = regressor.predict(X_test1)



x1 = raw_data['house area in 1000 square feet']
x2=raw_data['#bedrooms']
x3=raw_data['1 if condo, 0 otherwise ']
x4=raw_data['location ']

f1=np.ones(774)
f1=pd.Series(f1)
f2= x1
f3=[]
for i in range(len(f1)):
    if f2[i]>1500:
        n=f2[i]-1500
        f3.append(n)
    else:
        f3.append(0)
f3=pd.Series(f3)
f4= x2
f5=x3
f6=[]
f7=[]
f8=[]
for i in x4:    
    if i==1:
        f6.append(0)
        f7.append(0)
        f8.append(0)    
    elif i==2:
        f6.append(1)
        f7.append(0)
        f8.append(0) 
    elif i==3:
        f6.append(0)
        f7.append(1)
        f8.append(0)
    else:
        f6.append(0)
        f7.append(0)
        f8.append(1)
        
f6 = pd.Series(f6)
f7 = pd.Series(f7)
f8 = pd.Series(f8)

#new_y=raw_data[['selling price in 1000 dollars']]
new_model={"f1":f1,"f2":f2,"f3":f3,"f4":f4,"f5":f5,"f6":f6,"f7":f7,"f8":f8}
new_model_data = pd.DataFrame(new_model)

X_new = new_model_data
y_new =raw_data[['selling price in 1000 dollars']]

X_train_new, X_test_new, y_train_new, y_test_new= train_test_split(X_new, y_new, test_size=79, random_state=42)



new_regressor = LinearRegression() # instantiate the Linear Regression model
new_regressor.fit(X_train_new, y_train_new)

coef_model2=new_regressor.coef_
y_pred_new = new_regressor.predict(X_new)
# Creating the scatter plot for the new model

plt.scatter(y_new, y_pred_new, c='blue', label='Actual Price')
plt.scatter(y,y_pred, c='purple', label='Predicted Price')
plt.xlabel('Actual Sale Price') 
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs. Predicted Sale Prices')
plt.legend()
plt.show()

# number of folds
num_folds = 10

# Initialize an empty list to store the prediction errors
prediction_errors = []

# Perform cross-validation
kf = KFold(n_splits=num_folds)


cv_scores_simple = cross_val_score(regressor, X, y, cv=10, scoring='neg_mean_squared_error')
rmse_cv_model1 = np.sqrt(-cv_scores_simple.mean())

# Perform cross-validation for the complex model
cv_scores_complex = cross_val_score(new_regressor, X_new, y_new, cv=10, scoring='neg_mean_squared_error')
rmse_cv_model2 = np.sqrt(-cv_scores_complex.mean())

print("Cross-Validation RMSE for model 1:", rmse_cv_model1)
print("Cross-Validation RMSE for model 2:", rmse_cv_model2)



