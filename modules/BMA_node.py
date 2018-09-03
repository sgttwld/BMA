import numpy as np
from BMA_support import *
from BMA_agent import *
try:
    from scipy.special import lambertw
except:
    print("could not import lambertw (bounded priors won't work)")

class Node(object):

    def __init__(self,name='',dims=[],inds=[],num=1,cp=False):
        self.name = name
        self.ag = [Agent() for i in range(0,count(num))]
        # properties that are calculated by the node:
        self.marg = Dist()
        if cp:
            self.prior = Dist()
        else:
            self.prior = self.marg
        self.post = Dist()
        self.DKL = 0
        self.DKLpr = 0
        # properties that have to be defined in the system:
        self.p_in = Dist()   # used in marginal
        self.p0 = Dist()
        self.inds = inds
        self.beta_r = []
        self.dims = dims
        self.cp = cp

    def initialize(self):
        if len(self.inds) < 3:      # if no index for beta given, pick [0] as default (->len(beta)=1)
            self.inds.append([0])
        self.DKL = 0
        self.DKLpr = 0
        self.beta_r = self.inds[2]
        self.post.r = self.inds[1]
        self.p_in.r = self.post.r[:-1]
        self.post.initialize(self.dims)
        self.marg.r = self.inds[0]
        self.marg.initialize(self.dims)
        self.prior.r = self.inds[0]
        self.prior.initialize(self.dims)
        self.p0.r = self.inds[0]
        self.p0.val = normalize(np.ones(np.shape(self.prior.val)))
        for agent in self.ag: agent.reset()

    def update_input(self,joint):
        Z = np.einsum(joint.val,joint.r,self.prior.r[:-1])
        self.p_in.val = np.einsum(1.0/(Z+1e-55),self.prior.r[:-1],joint.val,joint.r,self.post.r[:-1])

    def update_posterior(self,U,beta):
        if np.shape(U) != np.shape(self.post.val):
            print("The utility must have the same shape as the posterior!")
        betatimesU = np.einsum(beta,self.beta_r,U,self.post.r,self.post.r)
        post = np.einsum(self.prior.val,self.prior.r,np.exp(betatimesU),self.post.r,self.post.r)
        self.post.val = normalize(post)

    def update_prior(self,alpha,beta):
        self.update_marginal()
        if self.cp: self.update_bounded_prior(alpha,beta)

    def update_marginal(self):
        self.marg.val = np.einsum(self.p_in.val,self.p_in.r,self.post.val,self.post.r,self.prior.r)

    def update_bounded_prior(self,alpha,beta):
        pr = np.copy(self.prior.val)
        if len(self.ag) > 1:
            for k in range(0,len(self.ag)):
                index = np.unravel_index(k,[self.dims[i] for i in self.beta_r])
                if alpha[index]/beta[index] > 500:
                    pr[index] = self.marg.val[index]/beta[index] - self.prior.val[index]*np.log(self.prior.val[index]/self.p0.val[index])/alpha[index]
                else:
                    DKL_pr = np.log(self.prior.val[index]/self.p0.val[index]).dot(self.prior.val[index])
                    cnst = alpha[index]/beta[index] - DKL_pr
                    denom = np.real(lambertw(np.exp(cnst)*(alpha[index]/beta[index])*self.marg.val[index]/self.p0.val[index]))
                    pr[index] = (alpha[index]/beta[index])*self.marg.val[index]/denom + 1e-55
        elif len(self.ag) == 1:
            if alpha[0]/beta[0] > 500:
                pr = self.marg.val/beta[0]-self.prior.val*np.log(self.prior.val/self.p0.val)/alpha[0]
            else:
                DKL_pr = np.log(self.prior.val/self.p0.val).dot(self.prior.val)
                cnst = alpha[0]/beta[0] - DKL_pr
                denom = np.real(lambertw(np.exp(cnst)*(alpha[0]/beta[0])*self.marg.val/self.p0.val)) + 1e-55
                pr = (alpha[0]/beta[0])*self.marg.val/denom + 1e-55
        self.prior.val = normalize(pr)


    def process(self,U,beta,alpha,joint):
        self.update_input(joint)
        self.update_posterior(U,beta)
        self.update_prior(alpha,beta)

    def calc_DKL(self):
        self.DKL = get_DKL(self.post,self.prior)

    def calc_DKLpr(self):
        self.DKLpr = get_DKL(self.prior,self.p0)

    def extract_agents(self):
        num = len(self.ag)
        if num > 1:
            ind = self.prior.r[:-1]    # indices of the dimensions that count this nodes agents
            dimind = [self.dims[ndx] for ndx in ind]
            rgoal = self.post.r[:]
            for i in ind:
                rgoal.remove(i)
            for k in range(0,num):
                delta = np.zeros(dimind)
                index = np.unravel_index(k,dimind)
                delta[index] = 1
                self.ag[k].post = np.einsum(delta,ind,self.post.val,self.post.r,rgoal)
                prior_r_ag = self.prior.r[:]
                for i in ind:
                    prior_r_ag.remove(i)
                self.ag[k].prior = np.einsum(delta,ind,self.prior.val,self.prior.r,prior_r_ag)
                pin_r_ag = self.p_in.r[:]
                for i in ind:
                    pin_r_ag.remove(i)
                self.ag[k].p_in = np.einsum(delta,ind,self.p_in.val,self.p_in.r,pin_r_ag)
                self.ag[k].calc_DKL()

        else:
            self.ag[0].post = self.post.val
            self.ag[0].prior = self.prior.val
            self.ag[0].p_in = self.p_in.val
            self.ag[0].calc_DKL()


## %%
