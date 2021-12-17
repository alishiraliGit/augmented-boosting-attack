import abc
import numpy as np


def zero_one_loss(y_pr, y):
    return np.mean((y_pr - y)**2)


class Evaluator(abc.ABC):
    def __init__(self, y, loss_fun):
        self._y = y
        self.n_sample = len(y)
        self.loss_fun = loss_fun

    def actual_risk(self, y_pr):
        return self.loss_fun(y_pr, self._y)

    @abc.abstractmethod
    def risk(self, y_pr):
        pass

    @abc.abstractmethod
    def to_string(self):
        pass


class Ladder(Evaluator):
    def __init__(self, y, eta, loss_fun=zero_one_loss):
        super().__init__(y, loss_fun)

        self.eta = eta
        self.R_min = np.inf
        
    @staticmethod
    def eta_round(val, eta):
        mod = val % eta
        r = 0
        if mod > (eta - mod):
            r = eta
        return (val // eta)*eta + r
        
    def risk(self, y_pr):
        R_pr = self.actual_risk(y_pr)
        if R_pr < self.R_min - self.eta:
            self.R_min = self.eta_round(R_pr, self.eta)
            
        return self.R_min

    def to_string(self):
        return r'Ladder($\eta=%.3f$)' % self.eta


class Kaggle(Evaluator):
    def __init__(self, y, decimals, loss_fun=zero_one_loss):
        super().__init__(y, loss_fun)

        self.decimals = decimals
        
    def risk(self, y_pr):
        return np.round(self.actual_risk(y_pr), decimals=self.decimals)

    def to_string(self):
        return r'Kaggle($decimals=%d$)' % self.decimals
