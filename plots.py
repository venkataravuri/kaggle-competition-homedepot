import numpy as np
import matplotlib.pyplot as plt


###############################################################################
# Plot feature importance
##############################################################################
def plot_feature_importance(feature_names, feature_importance):
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    sorted_feature_names = []
    for i in sorted_idx:
        sorted_feature_names.append(feature_names[i])
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, sorted_feature_names)
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()
