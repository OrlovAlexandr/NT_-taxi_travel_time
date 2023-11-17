import math

import numpy as np
from scipy.stats import f
from sklearn.linear_model import LinearRegression


def forward_feature_selection(factors, response, sig_level=0.05,
                              verbose=0, number=25):
    """
    Perform forward feature selection using linear regression.

    Parameters:
    - factors (DataFrame): The input factors (features).
    - response (Series): The response variable.
    - sig_level (float): The significance level for feature inclusion.
    - verbose (int): Verbosity level for printing details during selection.
    - number (int): Maximum number of iterations for feature selection.

    Returns:
    - List[str]: List of selected features.
    """

    # Initialize the linear regression model
    model = LinearRegression()

    # Get the sample size and the number of factors
    sample_size = len(response)
    factors_number = len(factors.columns)

    # Initialize lists to track selected and candidate features
    factors_to_use = np.arange(factors_number).tolist()
    factors_in_model = []
    factors_to_test = []

    # Calculate the restricted sum of squared errors (SSE)
    restricted_sse = np.sum(np.square(response - response.mean()))

    # Counter to control the number of iterations
    counter = 0

    flag = True
    while flag:
        # Initialize SSE for all factors to infinity
        full_sse = [math.inf] * factors_number

        # Iterate over factors to find the best candidate
        for i in factors_to_use:
            factors_to_test = factors_in_model[:]
            factors_to_test.append(i)
            model.fit(factors.iloc[:, factors_to_test], response)
            predict = model.predict(factors.iloc[:, factors_to_test])
            full_sse[i] = np.sum(np.square(predict - response))

        # Select the factor with the minimum SSE
        factor_candidate = np.argmin(full_sse)

        # Calculate the F-statistic and p-value for the selected factor
        df_rmf = 1
        df_full = sample_size - len(factors_to_test) - 1
        f_stat = ((restricted_sse - full_sse[factor_candidate]) / df_rmf) / \
                 (full_sse[factor_candidate] / df_full)

        p_value = 1 - f.cdf(f_stat, df_rmf, df_full)

        # Check if the p-value is below the significance level
        if p_value < sig_level:
            # Update the restricted SSE and add the factor to the model
            restricted_sse = full_sse[factor_candidate]
            factors_in_model.append(factor_candidate)
            factors_to_use.remove(factor_candidate)
        else:
            # Stop the loop if p-value is above the significance level
            flag = False

        # Print details if verbose mode is enabled
        if verbose != 0:
            print('Factor candidate:', factors.columns[factor_candidate])
            print('Full SSE:', np.around(full_sse, 2).tolist())
            print('F:', round(f_stat, 4))
            print('p-value:', p_value)
            print('Factors in model:', list(factors.columns[factors_in_model]))
            print('-' * 79 + '\n')

        # Check stopping conditions
        if len(factors_to_test) == 0:
            flag = False
            print('All factors are included')

        if counter >= number:
            flag = False

        counter += 1

    # Return the list of selected features
    return list(factors.columns[factors_in_model])
