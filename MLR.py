from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import statsmodels.api as sm

Stock_Market = {'Year':[2018,2018,2018,2018,2018,2018,2018,2018,2018,2018,2018,2018,\
                         2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,\
                         2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016],\
                'Month':[12,11,10,9,8,7,6,5,4,3,2,1, 12,11,10,9,8,7,6,5,4,3,2,1, 12,11,10,9,8,7,6,5,4,3,2,1],\
                'Interest_Rate':[2.73, 2.7,  2.63, 2.4,  2.37, 2.77, 2.53, 2.23, 2.67, 2.6,  2.43, 2.2,  2.5,  2.47,\
                                 2.03, 2.33, 2.3,  2.57, 2.27, 2.1,  2.,   2.13, 2.07, 2.17, 1.87, 1.73, 1.67, 1.97,\
                                 1.93, 1.9,  1.83, 1.8,  1.77, 1.6,  1.7,  1.63],\
                'Unemployment_Rate':[5.8,  5.75, 5.68, 5.65, 5.63, 5.62, 5.72, 5.83, 5.78, 5.73, 5.7,  5.6,  6.08, 5.67,\
                                     5.97, 5.95, 5.88, 5.87, 5.85, 5.82, 5.77, 5.98, 5.93, 5.92, 5.9,  6.07, 6.05, 6.,\
                                     6.03, 6.02, 6.15, 6.18, 6.17, 6.12, 6.13, 6.1],\
                'GDP':[22.2, 20.6, 21.7, 19.7, 22.1, 19.5, 18.7, 22.,  21.6, 19.4, 21.9, 21.8,\
                       21.5, 21.2, 21.1, 21.,  20.5, 21.4, 20.9, 20.4, 20.,  20.8, 20.3, 20.1,\
                       19.8, 19.6, 20.2, 19.3, 20.7, 18.9, 21.3, 19.2, 19.,  18.8, 19.9, 19.1],\
                'Stock_Index_Price': [1464,1394,1357,1293,1256,1254,1234,1230,1195,1175,1167,1159,\
                                      1147,1130,1075,1071,1065,1058,1051,1049,1043, 984, 976, 971,\
                                       968, 965, 958, 949, 943, 922, 884, 876, 866, 822, 719, 704]
                }

df = DataFrame(Stock_Market, columns=['Year','Interest_Rate','Unemployment_Rate','GDP', 'Stock_Index_Price'])
print(df.describe())

plt.scatter(df['Year'], df['Stock_Index_Price'], color='red')
plt.title('Stock Index Price Vs Year', fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Stock Index Price', fontsize=14)
plt.grid(True)
plt.show()

plt.scatter(df['Interest_Rate'], df['Stock_Index_Price'], color='red')
plt.title('Stock Index Price Vs Interest Rate', fontsize=14)
plt.xlabel('Interest Rate', fontsize=14)
plt.ylabel('Stock Index Price', fontsize=14)
plt.grid(True)
plt.show()

plt.scatter(df['Unemployment_Rate'], df['Stock_Index_Price'], color='red')
plt.title('Stock Index Price Vs Unemployement Rate', fontsize=14)
plt.xlabel('Unemployement Rate', fontsize=14)
plt.ylabel('Stock Index Price', fontsize=14)
plt.grid(True)
plt.show()

plt.scatter(df['Interest_Rate'], df['GDP'], color='red')
plt.title('Stock Index Price Vs GDP', fontsize=14)
plt.xlabel('GDP', fontsize=14)
plt.ylabel('Stock Index Price', fontsize=14)
plt.grid(True)
plt.show()


def MLR(X,Y):
  regr = linear_model.LinearRegression()
  regr.fit(X, Y)
  yhat= regr.predict(X)

  return regr.coef_, regr.intercept_, r2_score(Y, yhat)

results= []

results.append(MLR(df[['Interest_Rate']], df['Stock_Index_Price']))
results.append(MLR(df[['Unemployment_Rate']], df['Stock_Index_Price']))
results.append(MLR(df[['GDP']], df['Stock_Index_Price']))
results.append(MLR((df[['Interest_Rate', 'Unemployment_Rate']]), df['Stock_Index_Price']))
results.append(MLR((df[['Interest_Rate','GDP']]), df['Stock_Index_Price']))
results.append(MLR((df[['Unemployment_Rate','GDP']]), df['Stock_Index_Price']))
results.append(MLR((df[['Interest_Rate','Unemployment_Rate','GDP']]), df['Stock_Index_Price']))

for i, row in enumerate(results):
    print(f"Output {i+1}: coef is {row[0]}, int is {row[1]}, r2 is {row[2]}")

print("\n4 and 7 are the top models\n")

X4 = df[['Interest_Rate', 'Unemployment_Rate']]
X7 = df[['Interest_Rate', 'Unemployment_Rate', 'GDP']]
y = df['Stock_Index_Price']

model4 = linear_model.LinearRegression()
model4.fit(X4, y)

model7 = linear_model.LinearRegression()
model7.fit(X7, y)

# Test case 1
test1_model4 = DataFrame([[2.75, 5.3]],
                           columns=['Interest_Rate', 'Unemployment_Rate'])
test1_model7 = DataFrame([[2.75, 5.3, 20.0]],
                           columns=['Interest_Rate', 'Unemployment_Rate', 'GDP'])

# Test case 2
test2_model4 = DataFrame([[2.9, 5.9]],
                           columns=['Interest_Rate', 'Unemployment_Rate'])
test2_model7 = DataFrame([[2.9, 5.9, 21.0]],
                           columns=['Interest_Rate', 'Unemployment_Rate', 'GDP'])

print("Model 4 Predictions:")
print("Test 1:", model4.predict(test1_model4)[0])
print("Test 2:", model4.predict(test2_model4)[0])

print("\nModel 7 Predictions:")
print("Test 1:", model7.predict(test1_model7)[0])
print("Test 2:", model7.predict(test2_model7)[0])