import numpy as np


def basic_eda(df):

    rows = df.shape[0]
    cols = df.shape[1]
    null_values = df.isnull().sum().sum()

    return rows, cols, null_values


def detect_outliers_iqr(df):
    """Detect outliers in numeric columns using IQR method.
    Returns a dict: {col_name: {'count': N, 'lower': L, 'upper': U}}
    """

    results = {}
    numeric_df = df.select_dtypes(include=np.number)

    for col in numeric_df.columns:
        Q1 = numeric_df[col].quantile(0.25)
        Q3 = numeric_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = numeric_df[(numeric_df[col] < lower) | (numeric_df[col] > upper)]

        results[col] = {
            "count": len(outliers),
            "lower_bound": round(lower, 2),
            "upper_bound": round(upper, 2),
            "min": round(numeric_df[col].min(), 2),
            "max": round(numeric_df[col].max(), 2),
        }

    return results