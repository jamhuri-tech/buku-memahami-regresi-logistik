"""Bab 10: RegresiLogistikKita, regresi logistik biner dari nol.

Fungsi yang diminimumkan sama dengan LogisticRegression scikit-learn:
    1/2 ||w||^2 + C sum_i s_i loss_i          (penalti="l2")
    sum_i s_i loss_i                          (penalti=None)
dengan s_i bobot sampel dan intersep yang tidak dipenalti. Pilihan
penalti_intersep=True meniru liblinear: kolom konstan bernilai
skala_intersep ikut dipenalti seperti bobot lain.
Solvernya Newton teredam (Bab 9) pada fungsi di atas.
"""
import numpy as np
from scipy.special import expit


class RegresiLogistikKita:
    def __init__(self, C=1.0, penalti="l2", tol=1e-10,
                 max_iter=100, penalti_intersep=False,
                 skala_intersep=1.0):
        self.C, self.penalti = C, penalti
        self.tol, self.max_iter = tol, max_iter
        self.penalti_intersep = penalti_intersep
        self.skala_intersep = skala_intersep

    def _rancang(self, X):
        kolom = np.full(len(X), self.skala_intersep)
        return np.column_stack([kolom, X])

    def _bagian(self, theta, X, y, s):
        """Fungsi tujuan, gradien, dan Hessian di theta."""
        z = X @ theta
        p = expit(z)
        c = 1.0 if self.penalti is None else self.C
        r = np.ones(len(theta))                # bobot penalti
        r[0] = 1.0 if self.penalti_intersep else 0.0
        if self.penalti is None:
            r[:] = 0.0
        loss = np.logaddexp(0, z) - y * z
        f = c * s @ loss + 0.5 * r @ theta**2
        g = c * X.T @ (s * (p - y)) + r * theta
        H = c * (X.T * (s * p * (1 - p))) @ X + np.diag(r)
        return f, g, H

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        self.classes_ = np.unique(y)
        y = (np.asarray(y) == self.classes_[1]).astype(float)
        s = (np.ones(len(y)) if sample_weight is None
             else np.asarray(sample_weight, dtype=float))
        Xt = self._rancang(X)
        theta = np.zeros(Xt.shape[1])
        for k in range(1, self.max_iter + 1):
            f, g, H = self._bagian(theta, Xt, y, s)
            arah = np.linalg.solve(H, g)
            t, j = 1.0, g @ arah / 4
            while (self._bagian(theta - t * arah, Xt, y, s)[0]
                   > f - t * j):
                t /= 2
            theta = theta - t * arah
            g = self._bagian(theta, Xt, y, s)[1]
            if np.max(np.abs(g)) < self.tol:
                break
        self.n_iter_ = np.array([k])
        b = theta[0] * self.skala_intersep
        self.intercept_ = np.array([b])
        self.coef_ = theta[1:][None, :]
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_[0] + self.intercept_[0]

    def predict_proba(self, X):
        p = expit(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        positif = self.decision_function(X) > 0
        return self.classes_[positif.astype(int)]
