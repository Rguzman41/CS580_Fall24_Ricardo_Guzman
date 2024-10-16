import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# read data from a CSV file 
data = pd.read_csv('linear_regression_data.csv', header=None)


# separate the data into x (independent)  and y (depednent) variables
x = data[0].values
y = data[1].values

# calculate means of x and y
mean_x = np.mean(x)
mean_y = np.mean(y)

# calculate the covariance and variance
cov_xy = np.sum((x - mean_x) * (y - mean_y))
var_x = np.sum((x - mean_x) ** 2)

# calculate the slope and intercept of the linear regression line
b1 = cov_xy / var_x
b0 = mean_y - b1 * mean_x


# predicted values of y using the linear model
y_pred = b0 + b1 * x


# plot the data points and the regression line
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, y_pred, color='red', label='Regression line')
plt.xlabel('Independent variable (x)')
plt.ylabel('Dependent variable (y)')
plt.title('Linear Regression')
plt.legend()


#save the plot as an image
plt.savefig('linear_regression_plot.png')


# output the linear model parameters
print(f"Intercept (b0): {b0}")
print(f"Slope (b1): {b1}")


