import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from ipywidgets import interact, fixed, IntSlider


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def threshold_prediction(threshold):
    # Preprocessing
    threshold /= 100
    X_threhold = np.log(threshold / (1 - threshold))
    x_sigmoid = np.arange(-10, 10, 0.001)
    y_sigmoid = sigmoid(x_sigmoid)
    X = np.array([-7.35, 6.1, 0.5, -2.1, 4.02, -1.5, 2.01, 1.1, -0.5])
    y = np.array([0, 1, 0, 0, 1, 1, 1, 0, 1])
    y_pred = np.array([0 if x < X_threhold else 1 for x in X])

    # False Positives
    fp_idx = np.logical_and(y_pred == 1, y == 0)
    x_fp = X[fp_idx]
    y_fp = y[fp_idx]

    # False Negatives
    fn_idx = np.logical_and(y_pred == 0, y == 1)
    x_fn = X[fn_idx]
    y_fn = y[fn_idx]

    # True Positives
    tp_idx = np.logical_and(y_pred == 1, y == 1)
    x_tp = X[tp_idx]
    y_tp = y[tp_idx]

    # True Negatives
    tn_idx = np.logical_and(y_pred == 0, y == 0)
    x_tn = X[tn_idx]
    y_tn = y[tn_idx]

    # Plot
    plt.title('Threshold: {:.3f}\nAccuracy: {:.3f}\nPrecision: {:.3f}\nRecall: {:.3f}'.format(threshold,
                                                                                              accuracy_score(y, y_pred),
                                                                                              precision_score(y, y_pred),
                                                                                              recall_score(y, y_pred)),
              ha='left', x=-0)
    plt.plot(x_sigmoid, y_sigmoid, '-', lw=2, label='Sigmoid Function')
    plt.plot([-10, 10], [threshold, threshold], lw=5)
    plt.plot([X_threhold, X_threhold], [0, 1], lw=5)
    plt.plot(X_threhold, threshold, 'o', ms=16)
    plt.plot(X, y, 'o', ms=10, markerfacecolor='none', markeredgewidth=2)
    plt.plot(x_fp, y_fp, 'o', ms=9, label='False Positive')
    plt.plot(x_fn, y_fn, 'o', ms=9, label='False Negative')
    plt.plot(x_tp, y_tp, 'o', ms=9, label='True Positive')
    plt.plot(x_tn, y_tn, 'o', ms=9, label='True Negative')

    plt.xlabel('X, "body" word count', fontsize=16)
    plt.ylabel(r"$P(Y = SPAM | X)$", fontsize=16)
    plt.legend(loc=4)

    plt.show()


def threshold_prediction_plot(start=0.5):
    interact(threshold_prediction,
             threshold=IntSlider(value=start*100, min=0, max=100, step=1));
