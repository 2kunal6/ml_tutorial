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
def print_rsq_and_rsq_adj(data, y):
    cols = []
    for i in range(len(data[0])):
        cols.append(f'x{str(i)}')
    df = pd.DataFrame.from_records(data, columns=[cols])
    print(df)
    result = sm.OLS(y, df).fit()

    print(result.rsquared)
    print(result.rsquared_adj)


data = [[1, 11, 21, 31],
        [2, 12, 22, 32],
        [3, 13, 23, 33],
        [4, 14, 24, 34],
        [5, 15, 25, 35],
        [6, 16, 26, 36],
        [7, 17, 27, 37],
        [8, 18, 28, 38],
        [9, 19, 29, 39],
        [10, 20, 30, 40]]
y = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]


print_rsq_and_rsq_adj(data, y)

for i in range(len(data)):
    data[i].append(10-i)

print_rsq_and_rsq_adj(data, y)

for i in range(len(data)):
    data[i].append(i)

print_rsq_and_rsq_adj(data, y)
