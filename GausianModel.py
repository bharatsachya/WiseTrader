# gmm_em_from_scratch.py

import numpy as np

class GaussianMixtureEM:
    def __init__(self, n_components=3, max_iters=100, tol=1e-4):
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol

    def _initialize(self, X):
        n_samples, n_features = X.shape

        # Randomly initialize means
        rng = np.random.default_rng()
        self.means = X[rng.choice(n_samples, self.n_components, replace=False)]

        # Initialize covariances as identity matrices
        self.covariances = np.array([np.eye(n_features) for _ in range(self.n_components)])

        # Initialize mixing coefficients uniformly
        self.weights = np.ones(self.n_components) / self.n_components

    def _gaussian_pdf(self, X, mean, cov):
        n_features = X.shape[1]
        cov_det = np.linalg.det(cov)
        cov_inv = np.linalg.inv(cov)

        norm_const = 1.0 / np.sqrt((2 * np.pi) ** n_features * cov_det)
        diff = X - mean

        exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
        return norm_const * np.exp(exponent)

    def _e_step(self, X):
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            responsibilities[:, k] = self.weights[k] * self._gaussian_pdf(
                X, self.means[k], self.covariances[k]
            )

        # Normalize
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def _m_step(self, X, responsibilities):
        n_samples, n_features = X.shape
        Nk = responsibilities.sum(axis=0)

        for k in range(self.n_components):
            # Update mean
            self.means[k] = (responsibilities[:, k][:, np.newaxis] * X).sum(axis=0) / Nk[k]

            # Update covariance
            diff = X - self.means[k]
            self.covariances[k] = (
                responsibilities[:, k][:, np.newaxis] * diff
            ).T @ diff / Nk[k]

            # Add small value for numerical stability
            self.covariances[k] += 1e-6 * np.eye(n_features)

        # Update weights
        self.weights = Nk / n_samples

    def _log_likelihood(self, X):
        total = 0
        for k in range(self.n_components):
            total += self.weights[k] * self._gaussian_pdf(
                X, self.means[k], self.covariances[k]
            )
        return np.sum(np.log(total + 1e-10))

    def fit(self, X):
        self._initialize(X)
        prev_likelihood = None

        for i in range(self.max_iters):
            responsibilities = self._e_step(X)
            self._m_step(X, responsibilities)

            likelihood = self._log_likelihood(X)

            if prev_likelihood is not None:
                if abs(likelihood - prev_likelihood) < self.tol:
                    print(f"Converged at iteration {i}")
                    break

            prev_likelihood = likelihood

    def predict(self, X):
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)


# Example usage
if __name__ == "__main__":
    np.random.seed(42)

    # Generate synthetic data
    X1 = np.random.randn(100, 2) + np.array([0, 0])
    X2 = np.random.randn(100, 2) + np.array([5, 5])
    X3 = np.random.randn(100, 2) + np.array([0, 5])

    X = np.vstack([X1, X2, X3])

    model = GaussianMixtureEM(n_components=3, max_iters=100)
    model.fit(X)

    labels = model.predict(X)
    print("Cluster assignments:", labels)
