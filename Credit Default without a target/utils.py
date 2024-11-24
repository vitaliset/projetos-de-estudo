# https://github.com/dihanster/datalib/issues/17
# https://github.com/dihanster/datalib/pull/26
# https://github.com/dihanster/datalib/blob/main/datalib/metrics/_ranking.py

import numpy as np
import pandas as pd

from sklearn.utils.validation import _check_sample_weight
from pandas._libs.lib import is_integer

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics._base import (
    _average_binary_score,
    _average_multiclass_ovo_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils import (
    assert_all_finite,
    check_array,
    check_consistent_length,
    column_or_1d,
)
from sklearn.utils._encode import _encode, _unique
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import (
    _check_sample_weight,
    _check_pos_label_consistency,
)


def weighted_qcut(values, weights, q, **kwargs):
    # https://stackoverflow.com/questions/45528029/python-how-to-create-weighted-quantiles-in-pandas
    'Return weighted quantile cuts from a given series, values.'
    if is_integer(q):
        quantiles = np.linspace(0, 1, q + 1)
    else:
        quantiles = q
    order = pd.Series(weights).iloc[values.argsort()].cumsum()
    bins = pd.cut(order / order.iloc[-1], quantiles, **kwargs)
    return bins.sort_index()#.values

def discrete_lift_curve(y_true, y_pred, bins=10, sample_weight=None):
    sample_weight = _check_sample_weight(sample_weight, y_true)
    bins_scores = weighted_qcut(y_pred, sample_weight, bins, labels=False)
    return (np.array(list(range(bins))),
            np.array([np.average(y_true[bins_scores==q], weights=sample_weight[bins_scores==q]) for q in range(bins)]))
     
def last_bin_of_discrete_lift_curve(y_true, y_pred, bins=10, sample_weight=None):
    _, values = discrete_lift_curve(y_true, y_pred, bins=bins, sample_weight=sample_weight)
    return values[-1]
 
def mean_diff_of_discrete_lift_curve(y_true, y_pred, bins=10, sample_weight=None):
    _, values = discrete_lift_curve(y_true, y_pred, bins=bins, sample_weight=sample_weight)
    return pd.Series(values).diff().mean()


def numpy_fill(arr):
    """
    Solution provided by Divakar.
    https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array # noqa
    """
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = arr[idx]
    return out


def scipy_inspired_ks_2samp(data1, data2, wei1, wei2):
    ix1 = np.argsort(data1)
    ix2 = np.argsort(data2)

    min_data = np.min([data1[ix1[0]], data2[ix2[0]]])
    max_data = np.max([data1[ix1[-1]], data2[ix2[-1]]])

    data1 = np.hstack([min_data, data1[ix1], max_data])
    data2 = np.hstack([min_data, data2[ix2], max_data])
    wei1 = wei1[ix1]
    wei2 = wei2[ix2]
    data = np.sort(np.concatenate([data1, data2]))
    cwei1 = np.hstack([min_data, np.cumsum(wei1) / np.sum(wei1), max_data])
    cwei2 = np.hstack([min_data, np.cumsum(wei2) / np.sum(wei2), max_data])

    data = np.sort(np.concatenate([data1, data2]))
    distinct_value_indices = np.where(np.diff(data))[0]
    threshold_idxs = np.r_[distinct_value_indices, data.size - 1]

    dic1 = dict(zip(data1, cwei1))
    dic1.update({min_data: 0, max_data: 1})
    y1 = np.array(list(map(dic1.get, data[threshold_idxs])))
    y1 = numpy_fill(y1.astype(float))

    dic2 = dict(zip(data2, cwei2))
    dic2.update({min_data: 0, max_data: 1})
    y2 = np.array(list(map(dic2.get, data[threshold_idxs])))
    y2 = numpy_fill(y2.astype(float))

    return y1, y2, data[threshold_idxs]
    
def ks_curve(y_true, y_score, *, pos_label=None, sample_weight=None):
    # Check to make sure y_true is valid
    y_type = type_of_target(y_true, input_name="y_true")
    if not (
        y_type == "binary"
        or (y_type == "multiclass" and pos_label is not None)
    ):
        raise ValueError("{0} format is not supported".format(y_type))

    check_consistent_length(y_true, y_score, sample_weight)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    assert_all_finite(y_true)
    assert_all_finite(y_score)

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
        sample_weight = _check_sample_weight(sample_weight, y_true)
        nonzero_weight_mask = sample_weight != 0
        y_true = y_true[nonzero_weight_mask]
        y_score = y_score[nonzero_weight_mask]
        sample_weight = sample_weight[nonzero_weight_mask]

    pos_label = _check_pos_label_consistency(pos_label, y_true)

    # make y_true a boolean vector
    y_true = y_true == pos_label

    mask_pos = y_true == pos_label
    z1 = y_score[mask_pos]
    z0 = y_score[~mask_pos]

    if sample_weight is not None:
        w1 = sample_weight[mask_pos]
        w0 = sample_weight[~mask_pos]
    else:
        w1 = np.ones(z1.shape)
        w0 = np.ones(z0.shape)

    acum1, acum0, thresholds = scipy_inspired_ks_2samp(z1, z0, w1, w0)
    return acum1, acum0, thresholds

def delinquency_curve(y_true, y_score, pos_label=None, sample_weight=None):
    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError(
            f"Only binary classification is supported. Provided {labels}."
        )
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    check_consistent_length(y_true, y_score)
    pos_label = _check_pos_label_consistency(pos_label, y_true)
    assert_all_finite(y_true)
    assert_all_finite(y_score)

    y_true = y_true == pos_label

    scores_idxs = np.argsort(y_score)[::1]
    actual_idxs = np.argsort(y_true)[::1]

    weights = _check_sample_weight(
        sample_weight, y_true, only_non_negative=True
    )

    weights = weights / weights.max()
    y_true_sorted_by_scores = y_true[scores_idxs].copy()
    y_true_sorted = y_true[actual_idxs].copy()
    sorted_weights = weights[scores_idxs].copy()

    step_approval_rate = 1 / len(y_true_sorted_by_scores)
    approval_rate = np.arange(0, 1 + step_approval_rate, step_approval_rate)
    default_rate = np.append(
        0, y_true_sorted_by_scores.cumsum() / sorted_weights.cumsum()
    )
    optimal_rate = np.append(
        0, y_true_sorted.cumsum() / sorted_weights.cumsum()
    )

    return approval_rate, default_rate, optimal_rate
