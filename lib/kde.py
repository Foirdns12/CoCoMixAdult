import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from statsmodels.nonparametric.kernel_density import KDEMultivariate


def initialize_pdf(data, var_types, categorical_values, bw):
    """
    Initialize an estimator for the PDF of the provided *data*.

    :param data:
    :param var_types:
    :param categorical_values:
    :param bw:
    :return:
    """
    kde, preprocess = initialize_kde(data, var_types, categorical_values, bw)

    def compute_pdf(samples):
        samples = np.array(samples)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
            return kde.pdf(preprocess(samples))
        else:
            return kde.pdf(preprocess(samples))

    return compute_pdf


# noinspection PyPep8Naming
def initialize_kde(data, var_types, categorical_values, bw,
                   _KDE=KDEMultivariate, _scale_numerical_columns=True, **kwargs):
    """
    Initialize a multivariate KDE and a preprocessor on the provided *data*.

    :param data:
    :param var_types:
    :param categorical_values:
    :param bw:
    :param _KDE: Which KDE class to use
    :param _scale_numerical_columns: Whether to standard scale numerical columns
    :param kwargs: Keyword arguments to be passed to KDE
    :return: tuple of pdf and preprocessing function
    """
    var_type_str = "".join(var_types)
    preprocess = _create_preprocessor(data, var_type_str, categorical_values,
                                      _scale_numerical_columns)
    x = preprocess(data)
    kde = _KDE(x, var_type_str, bw=bw, **kwargs)

    return kde, preprocess


def _create_preprocessor(data, var_type_str, categorical_values="auto", _scale_numerical_columns=True,
                         categorical_encoder="ordinal"):
    print(var_type_str)
    print(categorical_values)
    if data.shape[1] != len(var_type_str):
        raise ValueError("Expect data in shape (n_samples, n_features)")

    if isinstance(categorical_values, list) and \
            (len(categorical_values) != len(var_type_str) - var_type_str.count("c")):
        raise ValueError(f"More categorical value arrays ({len(categorical_values)}) "
                         f"than categorical variables ({len(var_type_str) - var_type_str.count('c')})")

    categorical_features = [feature_idx for feature_idx, var_type in enumerate(var_type_str)
                            if var_type != "c"]
    if categorical_encoder == "ordinal":
        categorical_transformer = Pipeline(steps=[('ordinal', OrdinalEncoder(categories=categorical_values))])
    elif categorical_encoder == "onehot":
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder())])
    else:
        raise ValueError(f"Unknown categorical encoder '{categorical_encoder}."
                         f"Possible options are 'onehot' and 'ordinal'.")

    scalar_features = [feature_idx for feature_idx, var_type in enumerate(var_type_str)
                       if var_type == "c"]
    scalar_transformer = Pipeline(steps=[('scaler', StandardScaler(with_mean=_scale_numerical_columns,
                                                                   with_std=_scale_numerical_columns))])

    preprocessor = ColumnTransformer(transformers=[
        ("scalar", scalar_transformer, scalar_features),
        ("categorical", categorical_transformer, categorical_features)
    ])
    preprocessor.fit(data)

    # The ColumnTransformer sorts the column by type, i.e. data with
    # var_types 'uoccucoo' will be returned as 'cccuouoo', so we need
    # to re-sort the data
    column_map = scalar_features + categorical_features

    def preprocess(new_data):
        transformed_data = preprocessor.transform(new_data)
        transformed_data.T[column_map] = transformed_data.T[list(range(len(var_type_str)))]
        return transformed_data

    return preprocess


def _create_preprocessor2(data, var_type_str, categorical_values="auto", _scale_numerical_columns=True):
    return _create_preprocessor(data, var_type_str,
                                categorical_values=categorical_values,
                                _scale_numerical_columns=_scale_numerical_columns,
                                categorical_encoder="onehot")
