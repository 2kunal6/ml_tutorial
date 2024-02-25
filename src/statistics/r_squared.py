from sklearn.metrics import r2_score

import pandas as pd
import statsmodels.api as sm

def r2_score_custom(y, y_p):
    n = len(y)
    sum_square_regression = 0
    total_sum_of_squares = 0
    y_mean = (sum(y) / n)

    for i in range(n):
        sum_square_regression += ( (y[i]-y_p[i]) ** 2 )
        total_sum_of_squares += ( (y[i]-y_mean) ** 2 )

    return 1 - (sum_square_regression/total_sum_of_squares)



y =      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y_pred = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y_pred_2 = [1, 5, 3, 4, 6, 6, 7, 8, 9, 10.5]
y_pred_3 = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

print(r2_score(y, y_pred))
print(r2_score_custom(y, y_pred))

print(r2_score(y, y_pred_2))
print(r2_score_custom(y, y_pred_2))

print(r2_score(y, y_pred_3))
print(r2_score_custom(y, y_pred_3))


# Ajusted r2 score ------------------------------------------------------------

#Adj_r2 = 1 - (1-r2_score(y, y_pred)) * (len(y)-1)/(len(y)-X.shape[1]-1)
data = pd.read_csv('data.csv')
print(data[["x1", "x2"]])
#fit regression model
X, y = data[["x1", "x2", "x3", "x4"]], data.y
result = sm.OLS(y, X).fit()

print(result.summary())
