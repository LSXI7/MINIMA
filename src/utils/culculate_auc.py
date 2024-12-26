import numpy as np

# The provided function for calculating AUC
def error_auc(errors, thresholds):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}

# The provided error data and thresholds
errors = [4.3644, 3.2340, 4.4359, 1.3157, 2.9037, 2.9106, 3.6509, 0.0317, 1.8696, 3.6857]
thresholds = [1, 3, 5]

# Calculating AUC using the given function
auc_results = error_auc(errors, thresholds)
print(auc_results)
