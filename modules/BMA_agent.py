import numpy as np

class Agent(object):

    def __init__(self):
        self.prior = 0
        self.post = 0
        self.p_in = 0
        self.DKL = 0

    def reset(self):
        self.prior = 0
        self.post = 0

    def calc_DKL(self):
        eps = 1e-55
        r = list(range(0,len(np.shape(self.post)))) # indices for posterior
        rprior = r[:]
        for i in range(0,len(np.shape(self.post))):
            if len(np.shape(self.prior)) < len(rprior):
                del rprior[0]
        D = ( np.einsum(self.post,r,np.log(self.post + eps),r,r[:-1])
             -np.einsum(self.post,r,np.log(self.prior + eps),rprior,r[:-1]) )
        self.DKL = np.einsum(D,r[:-1],self.p_in,r[:-1],[])
