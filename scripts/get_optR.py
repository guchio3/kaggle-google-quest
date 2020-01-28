import pickle
from functools import partial
from glob import glob

import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import spearmanr


class OptimizedRounder(object):
    """
    An optimizer for rounding thresholds
    to maximize Quadratic Weighted Kappa (QWK) score
    # https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved
    """

    def __init__(self):
        self.coef_ = 0

    def _spearmanr_loss(self, coef, X, y, labels):
        """
        Get loss according to
        using current coefficients
        :param coef: A list of coefficients that will be used for rounding
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) +
                     [np.inf], labels=labels)

        # return -np.mean(spearmanr(y, X_p).correlation)
        return -spearmanr(y, X_p).correlation

    def fit(self, X, y, initial_coef):
        """
        Optimize rounding thresholds
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        labels = self.labels
        loss_partial = partial(self._spearmanr_loss, X=X, y=y, labels=labels)
        self.coef_ = sp.optimize.minimize(
            loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        """
        Make predictions with specified thresholds
        :param X: The raw predictions
        :param coef: A list of coefficients that will be used for rounding
        """
        labels = self.labels
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) +
                      [np.inf], labels=labels)
        # [np.inf], labels=[0, 1, 2, 3])

    def coefficients(self):
        """
        Return the optimized coefficients
        """
        return self.coef_['x']

    def set_labels(self, labels):
        self.labels = labels


def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        if len(np.unique(col_pred)) == 1:
            if col_pred[0] == np.max(col_trues):
                col_pred[np.argmin(
                    col_pred)] = np.min(col_trues)
            else:
                col_pred[np.argmax(
                    col_pred)] = np.max(col_trues)
        rhos.append(
            spearmanr(
                col_trues,
                col_pred
                #                  + np.random.normal(
                #                     0,
                #                     1e-7,
                #                     col_pred.shape[0])
            ).correlation)
    return rhos


def main():
    BASE_PATH = './mnt/checkpoints/e030/'

    y_preds = []
    y_trues = []
    for i in range(5):
        ckpt = glob(f'{BASE_PATH}/{i}/*.pth')[0]
        y_preds.append(ckpt['val_y_preds'])
        y_trues.append(ckpt['val_y_trues'])
    y_preds = np.conatenate(y_preds)
    y_trues = np.conatenate(y_trues)

    reses = []
    optRs = []

    for i in range(30):
        optR = OptimizedRounder()
        labels = np.sort(np.unique(y_trues[:, i]))
        optR.set_labels(labels)
        initial_coef = (labels[:-1] + labels[1:]) / 2
        optR.fit(y_preds[:, i], y_trues[:, i], initial_coef=initial_coef)
        optRs.append(optR)
        res = optR.predict(y_preds[:, i], optR.coefficients())
        reses.append(res)
    reses = np.asarray(reses).T

    with open(f'{BASE_PATH}/optRs.pkl', 'wb') as fout:
        pickle.dump(optRs, fout)

    res_score = compute_spearmanr(y_trues, reses)
    print(f'res_score: {res_score}')


if __name__ == '__main__':
    main()
