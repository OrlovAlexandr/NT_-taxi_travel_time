import math

import numpy as np
from scipy.stats import f
from sklearn.linear_model import LinearRegression


def forward_feature_selection(factors, response, sig_level=0.05, verbose=0, number=25):
    model = LinearRegression()

    sample_size = len(response)
    factors_number = len(factors.columns)
    factors_to_use = np.arange(factors_number).tolist()
    factors_in_model = []
    factors_to_test = []

    restricted_sse = np.sum(np.square(response - response.mean()))

    counter = 0

    flag = True
    while flag:
        full_sse = [math.inf] * factors_number
        for i in factors_to_use:
            factors_to_test = factors_in_model[:]
            factors_to_test.append(i)
            model.fit(factors.iloc[:, factors_to_test], response)
            predict = model.predict(factors.iloc[:, factors_to_test])
            full_sse[i] = np.sum(np.square(predict - response))

        factor_candidate = np.argmin(full_sse)

        df_rmf = 1
        df_full = sample_size - len(factors_to_test) - 1
        f_stat = ((restricted_sse - full_sse[factor_candidate]) / df_rmf) / (full_sse[factor_candidate] / df_full)

        p_value = 1 - f.cdf(f_stat, df_rmf, df_full)

        if p_value < sig_level:
            restricted_sse = full_sse[factor_candidate]
            factors_in_model.append(factor_candidate)
            factors_to_use.remove(factor_candidate)
        else:
            flag = False
        if verbose != 0:
            print('Factor candidate:', factors.columns[factor_candidate])
            print('Full SSE:', np.around(full_sse, 2).tolist())
            print('F:', round(f_stat, 4))
            print('p-value:', p_value)
            print('Factors in model:', list(factors.columns[factors_in_model]))
            print('-' * 79 + '\n')

        if len(factors_to_test) == 0:
            flag = False
            print('All are included')

        if counter >= number:
            flag = False

        counter += 1

    return list(factors.columns[factors_in_model])
