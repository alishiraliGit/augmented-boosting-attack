import numpy as np

rng = np.random.default_rng(1)


class KNNRecSys:
    def __init__(self, n_c_star, k_star):
        self.n_c_star = n_c_star  # num. of central points per class
        self.k_star = k_star
        self.centers_star, self.labels_star = self.init_ground_truth()

    def init_ground_truth(self):
        labels_star = np.array([0]*self.n_c_star + [1]*self.n_c_star)
        centers_star = rng.random(size=(labels_star.shape[0], 2))
        return centers_star, labels_star

    @staticmethod
    def score(u, i, centers, labels, k):
        v = np.array([[u, i]])
        top_labels = labels[np.argsort(np.sum((centers - v)**2, axis=1))[:k]]
        return (np.mean(top_labels) >= 0.5)*1

    def run(self, n_sample, k, exploration):
        # Init.
        centers = rng.random((n_sample, 2))
        labels = np.zeros((n_sample,))

        # Get score for first 2k samples (2k is just my choice for initial exploration!)
        labels[:2*k] = np.array([self.score(c[0], c[1],
                                            centers=self.centers_star,
                                            labels=self.labels_star,
                                            k=self.k_star)
                                 for c in centers[:2*k]])

        # Loop
        for idx in range(2*k, n_sample):
            # Select a random user
            u = rng.random()

            # Predict for items
            items = np.arange(0, 1, 1e-2)

            scores_pr = np.array([self.score(u, i,
                                             centers=centers[:idx],
                                             labels=labels[:idx], k=k)
                                  for i in items])

            # Select an item with positive prediction
            p = scores_pr*(1 - exploration) + exploration
            p = p/np.sum(p)
            i_chosen = rng.choice(items, p=p)

            # Get feedback
            score_te = self.score(u, i_chosen,
                                  centers=self.centers_star,
                                  labels=self.labels_star,
                                  k=self.k_star)

            # Update S and labels
            centers[idx] = np.array([u, i_chosen])
            labels[idx] = score_te

        return centers, labels
