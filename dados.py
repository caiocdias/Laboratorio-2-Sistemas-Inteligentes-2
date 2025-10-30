from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

X, _ = make_circles(n_samples=3000, factor=0.5, noise=0.05)
plt.scatter(X[:,0], X[:,1], s=2)
plt.axis('equal')
plt.title("Dataset - Círculos Concêntricos")
plt.show()

def make_circles(n_samples=2000, noise=0.05, seed=7):
    rng = np.random.default_rng(seed)
    n = n_samples // 2
    t = rng.uniform(0, 2*np.pi, n)
    outer = np.stack([np.cos(t), np.sin(t)], axis=1)
    inner = 0.5 * np.stack([np.cos(t), np.sin(t)], axis=1)
    X = np.vstack([outer, inner])
    X += rng.normal(0, noise, X.shape)
    return X