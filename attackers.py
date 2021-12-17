import abc
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation

from utils.math_tools import bits2, argmaxs
from evaluators import Evaluator, Kaggle
from simulators import KNNRecSys

rng = np.random.default_rng(1)


class Attacker(abc.ABC):
    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator

    @abc.abstractmethod
    def query_and_update(self, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self):
        pass

    @abc.abstractmethod
    def to_string(self):
        pass


class BoostingAttacker(Attacker):
    def __init__(self, evaluator: Evaluator, compare_to_min_loss=False):
        super().__init__(evaluator)

        # Spec
        self.n_sample = self.evaluator.n_sample
        self.compare_to_min_loss = compare_to_min_loss

        # State
        self.U = None
        self.Rs = []
        self.I = []  # indices of informative predictions
        self.R_min = np.inf  # smallest observed loss
        self.t = 0

    def _update_U(self, u):
        if self.U is None:
            self.U = u.reshape((1, -1))
        else:
            self.U = np.concatenate((self.U, u.reshape(1, -1)), axis=0)

    def _toss(self):
        return (rng.random((self.n_sample,)) > 0.5)*1

    def query_and_update(self):
        t = self.t

        # Randomly predict
        u = self._toss()

        # Update U
        self._update_U(u)

        # Evaluate the prediction
        R = self.evaluator.risk(u)

        # Update Rs and actual_Rs
        self.Rs.append(R)

        # Update I
        if (R < 0.5) and (not self.compare_to_min_loss or R < self.R_min):
            self.I.append(t)

        # Update minimum Rs
        self.R_min = np.minimum(self.R_min, R)

        # Forward time
        self.t += 1

    def predict(self):
        if len(self.I) == 0:
            return np.nan

        return (np.mean(self.U[self.I], axis=0) > 0.5)*1

    def to_string(self):
        return r'Boosting'


class AdaptiveRandomWindowSearchBoostingAttacker(Attacker):
    def __init__(self, evaluator: Evaluator, w, alpha, compare_to_min_loss=False):
        super().__init__(evaluator)

        # Spec
        self.n_sample = self.evaluator.n_sample
        self.w = w
        self.alpha = alpha
        self.compare_to_min_loss = compare_to_min_loss

        # State
        self.U = None
        self.Rs = []
        self.I = []  # indices of informative predictions
        self.R_min = np.inf  # smallest observed loss
        self.b = np.ones((self.n_sample,))*0.5
        self.t = 0

    def _update_U(self, u):
        if self.U is None:
            self.U = u.reshape((1, -1))
        else:
            self.U = np.concatenate((self.U, u.reshape(1, -1)), axis=0)

    def _toss(self):
        """
        Generates a random 0,1 vector. Samples are biased according to b.
        :return: (n_sample,)
        """
        return (rng.random((self.n_sample,)) <= self.b)*1

    def query_and_update(self):
        t = self.t

        # Randomly predict
        u = self._toss()

        if t % 2 == 1:
            w_start = np.random.randint(0, self.n_sample - 1 - self.w)
            w_end = w_start + self.w
            u[w_start:w_end] = 1 - (np.mean(self.U[t - 1, w_start:w_end]) > 0.5)*1

        # Update U
        self._update_U(u)

        # Evaluate the prediction
        R = self.evaluator.risk(u)

        # Update Rs and actual_Rs
        self.Rs.append(R)

        # Update I
        if (R < 0.5) and (not self.compare_to_min_loss or R < self.R_min):
            self.I.append(t)

        # Update minimum Rs
        self.R_min = np.minimum(self.R_min, R)

        # Update b
        if t % 2 == 1:
            dR = R - self.Rs[t - 1]
            dN = np.sum(u[w_start:w_end] - self.U[t - 1, w_start:w_end])
            b_est = (1 - self.n_sample*dR/dN)/2
            self.b[w_start:w_end] = (1 - self.alpha)*self.b[w_start:w_end] + self.alpha*b_est

        # Forward time
        self.t += 1

    def predict(self):
        if len(self.I) == 0:
            return np.nan

        return (np.mean(self.U[self.I], axis=0) > 0.5)*1

    def to_string(self):
        return r'WBoost($w=%d,\alpha=%.2f$)' % (self.w, self.alpha)


class CorrelatedBoostingAttacker(BoostingAttacker):
    def __init__(self, evaluator: Evaluator, gamma, compare_to_min_loss=False):
        super().__init__(evaluator, compare_to_min_loss)

        self.gamma = gamma
        self.cov = self.pow_cov(self.n_sample, self.gamma)

        # For performance issues
        self.U_future = None

    @staticmethod
    def pow_cov(n, gamma):
        cov = np.eye(n)
        for k in range(1, n):
            cov += np.diag(np.ones((n - k,)), k=k) * gamma**k
            cov += np.diag(np.ones((n - k,)), k=-k) * gamma**k
            if gamma**k < 1e-4:
                break
        return cov

    def _toss(self):
        if self.t % 100 == 0:
            self.U_future = \
                rng.multivariate_normal(mean=np.zeros((self.n_sample,)), cov=self.cov, size=(100,))

        u_norm = self.U_future[self.t % 100]
        return (u_norm > 0) * 1

    def to_string(self):
        return r'CorrBoost($\gamma=%.2f$)' % self.gamma


class KNNMAPBoostingAttacker(BoostingAttacker):
    def __init__(self, evaluator: Evaluator, centers, k, N, exploration, conf,
                 do_grouping=True, grouping_depth=1,
                 compare_to_min_loss=False,
                 verbose=False):
        super().__init__(evaluator, compare_to_min_loss)

        # Constants
        self.k = k
        self.N = N  # number of quantization points
        self.exploration = exploration
        self.conf = conf

        # Data
        self.centers = centers

        # Processed from data
        if do_grouping:
            self.groups = self._find_groups(grouping_depth=grouping_depth, verbose=verbose)
        else:
            self.groups = [list(range(self.n_sample))]

        self.g_map_est_dic = self.find_map_estimates_for_groups(verbose=verbose)
        self.g_top_dic = self.find_most_probables(verbose=verbose)

    @staticmethod
    def effective_set(u, centers, k, N):
        eff_set = set()
        for int_i in range(N):
            v = np.array([[u, int_i/N]])
            eff_set.update(np.argsort(np.sum((centers - v)**2, axis=1))[:k].tolist())
        return list(eff_set)

    def _find_groups(self, grouping_depth=1, verbose=False):
        assert grouping_depth <= 2

        if verbose:
            print('Finding groups ...')

        k, N = self.k, self.N
        candidate_nodes = set(range(self.n_sample))

        groups = []

        while len(candidate_nodes) > 0:
            candidate_nodes_np = np.array(list(candidate_nodes))

            # Select a random node
            node = rng.choice(candidate_nodes_np)

            # Init. a group
            group = {node, }

            # Update the group with 1st level effective nodes
            candidates_older = candidate_nodes_np[candidate_nodes_np < node]
            eff_nodes = candidates_older[self.effective_set(self.centers[node, 0],
                                                            self.centers[candidates_older],
                                                            k, N)]
            group.update(eff_nodes)

            # Update the group with 2nd level effective nodes
            if grouping_depth == 2:
                for eff_node in eff_nodes:
                    candidates_older = candidate_nodes_np[candidate_nodes_np < eff_node]
                    eff_eff_nodes = \
                        candidates_older[self.effective_set(self.centers[eff_node, 0],
                                                            self.centers[candidates_older],
                                                            k, N)]
                    group.update(eff_eff_nodes)

            # Update all groups
            groups.append(np.sort(list(group)))

            # Drop the group members from the candidates
            candidate_nodes -= group

        return groups

    @staticmethod
    def factor(node, centers, node_eff_dic, k, N, exploration):
        # Extract node's spec
        u, i = centers[node]
        eff_list = node_eff_dic[node]
        n_eff = len(eff_list)

        # Init.
        score_mat = np.zeros((2,)*n_eff + (N,))
        p_mat = np.zeros((2,)*n_eff)

        # Loop over all possible labels of the effective set
        for num in range(2**n_eff):
            labels = np.array(bits2(num, n_eff))

            # Loop over possible items (N quantized values)
            for int_i in range(N):
                score_mat[tuple(labels) + (int_i,)] = \
                    (1 - exploration)*KNNRecSys.score(u, int_i/N, centers[eff_list], labels, k) \
                    + exploration

            # Calc the probability of being selected
            score_sum = np.sum(score_mat[tuple(labels)])
            score_i = (1 - exploration)*KNNRecSys.score(u, i, centers[eff_list], labels, k) \
                + exploration
            if score_sum == 0:
                p_mat[tuple(labels)] = 1
            else:
                p_mat[tuple(labels)] = score_i/score_sum*N

        # Create a factor node
        phi = DiscreteFactor(variables=['dummy%d' % node] + ['n%d' % eff_node for eff_node in eff_list],
                             cardinality=[1] + [2]*n_eff,
                             values=p_mat.reshape((-1,)))

        return phi, score_mat, p_mat

    @staticmethod
    def map_estimate(centers, k, N, exploration, verbose=False):
        # centers' rows should be sorted in time
        n_sample = centers.shape[0]

        # Find effective nodes
        node_eff_dic = {}
        all_eff_nodes = set()
        for node in range(n_sample):
            u, i = centers[node]
            node_eff_dic[node] = KNNMAPBoostingAttacker.effective_set(u, centers[:node], k, N)
            all_eff_nodes.update(node_eff_dic[node])

        if verbose:
            print('There are totally %d effective nodes' % len(all_eff_nodes))

        # Init. a factor graph
        G = FactorGraph()

        # Add nodes
        dummy_nodes_str = ['dummy%d' % eff_node for eff_node in range(1, n_sample)]
        eff_nodes_str = ['n%d' % eff_node for eff_node in all_eff_nodes]
        G.add_nodes_from(dummy_nodes_str + eff_nodes_str)

        # Add a dummy factor (ToDo: this is my solution to no sepset exception)
        dummy_phi = DiscreteFactor(
            variables=dummy_nodes_str,
            cardinality=[1]*(n_sample - 1),
            values=np.ones((1,)))
        G.add_factors(dummy_phi)
        G.add_edges_from([(dummy_node_str, dummy_phi) for dummy_node_str in dummy_nodes_str])

        # Add factor nodes
        if verbose:
            print('Adding factor nodes ...')

        for node in tqdm(range(1, n_sample), disable=not verbose):
            phi, _, _ = KNNMAPBoostingAttacker.factor(node, centers, node_eff_dic, k, N, exploration)

            G.add_factors(phi)
            G.add_edges_from([('dummy%d' % node, phi)])
            G.add_edges_from([('n%d' % node_eff, phi) for node_eff in node_eff_dic[node]])

            assert G.number_of_nodes() == n_sample + len(all_eff_nodes) + node

        if verbose:
            print(G)
            print('Model checked: %s' % G.check_model())

        # Init. a BP
        if verbose:
            print('Init. BP ...')

        BP = BeliefPropagation(G)

        # Query BP
        q = BP.query(variables=['n%d' % eff_node for eff_node in all_eff_nodes], show_progress=verbose)

        # Process results
        variables = np.array([int(var[1:]) for var in q.variables])

        return variables, q.values

    def find_map_estimates_for_groups(self, verbose=False):
        if verbose:
            print('MAP estimation for each group ...')

        g_dic = {}
        for i_g, g in tqdm(enumerate(self.groups), disable=not verbose):
            if len(g) == 1:
                continue

            try:
                variables, values = self.map_estimate(self.centers[np.sort(g)],
                                                      self.k,
                                                      self.N,
                                                      self.exploration,
                                                      verbose=False)
                g_dic[i_g] = {
                    'vars': np.sort(g)[variables],
                    'vals': values
                }

            except Exception as e:
                print('For group %d:' % i_g, g)
                print(e)

        return g_dic

    def find_most_probables(self, verbose=False):
        if verbose:
            print('Finding most probable outcomes ...')

        g_top_dic = {}
        for i_g, dic in self.g_map_est_dic.items():
            rs, ps = argmaxs(dic['vals'], self.conf)
            g_top_dic[i_g] = {
                'vars': dic['vars'],
                'rs': rs,
                'ps': ps,
            }
        return g_top_dic

    def _toss(self):
        u = np.zeros((self.n_sample,))*np.nan

        for i_g, dic in self.g_top_dic.items():
            idx = rng.choice(range(dic['rs'].shape[0]), p=dic['ps']/np.sum(dic['ps']))
            u[dic['vars']] = dic['rs'][idx]

        # Fill nans
        nan_indices = np.argwhere(np.isnan(u))[:, 0]
        u[nan_indices] = np.random.randint(0, 2, size=(len(nan_indices),))

        return u.astype(int)

    def to_string(self):
        return r'MAP($k=%d, N=%d, expl=%.1f, conf=%0.2f$)' \
               % (self.k, self.N, self.exploration, self.conf)


if __name__ == '__main__':
    '''
    y_te = np.zeros((400,))
    y_te[200:] = 1

    att = AdaptiveRandomWindowSearchBoostingAttacker(Kaggle(y_te, decimals=5), w=21, alpha=0.1)

    n_q = 200
    risks = np.zeros((n_q,))
    for qu in range(n_q):
        att.query_and_update()
        y_pr = att.predict()
        risks[qu] = att.evaluator.actual_risk(y_pr)

    plt.subplot(2, 1, 1)
    plt.plot(att.b)
    plt.subplot(2, 1, 2)
    plt.plot(risks)
    plt.show()
    '''

    y_ = np.array([1, 1, 0])
    centers_ = np.array([[0, 0],
                         [1, 1],
                         [0.6, 0]])

    variables_, values_ = KNNMAPBoostingAttacker.map_estimate(centers_,
                                                              k=1, N=50, exploration=0.1, verbose=False)

    ev_ = Kaggle(y_, decimals=5)
    att_ = KNNMAPBoostingAttacker(ev_, centers_, k=1, N=50, exploration=0.1, conf=0.99, do_grouping=False,
                                  compare_to_min_loss=True, verbose=True)

    U_ = np.zeros((10000, 3)).astype(int)
    for sample_ in range(U_.shape[0]):
        U_[sample_] = att_._toss()

    print(np.mean(U_, axis=0))
