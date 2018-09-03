import numpy as np
from BMA_support import *
from BMA_node import *
from BMA_agent import *
from BMA_sysconfig import *

class MultiAgentSystem(object):

    def __init__(self,tp,shape):
        self.tp = tp
        self.U = 0
        self.dims = []
        self.pw = Dist()
        self.F_curr = 0
        self.pagw = 0
        self.EU = 0
        self.DKL = 0
        self.FE = 0
        self.joint = Dist()
        self.nd = []
        self.ag = []
        self.beta = []
        self.alpha = []
        self.cp = False
        self.shape = shape
        if self.tp==(1,2):
            if self.shape[1] != self.shape[2]:
                print("Error: A and X2 must have the same amount of agents!")
        self.M = []
        self.conf = BMA_sysconfig
        self.nd_lst = []
        self.cooling = False
        self.alpharange = []

    def initialize(self,U,beta=[],M=[],alpha=[],alphas=[]):
        self.U = U
        self.M = M
        self.set_dims(self.U,self.M)
        self.pw.val = np.ones(self.dims[0])/self.dims[0]
        self.pw.r = [0]
        self.joint.val = np.zeros(self.dims)
        self.joint.r = range(0,len(np.shape(self.joint.val)))
        self.nd_lst = gen_node_list(self.tp)
        self.nd = []
        self.ag = []
        if len(alphas)>0:
            self.cooling = True
        if len(alpha)>0 or self.cooling:
            self.cp = True
        for i in range(0,len(self.nd_lst)):
            name = self.nd_lst[i]
            self.nd.append(Node(name,self.dims,self.conf[self.tp]['nodes'][name],self.shape[i],self.cp))
        for node in self.nd: node.initialize()
        self.propagate_agents()
        self.F_curr = 0*np.ones((self.dims[0],self.dims[1])) + np.random.rand(self.dims[0],self.dims[1])
        if self.cp and not(self.cooling):
            self.alpha = reshape_list(alpha,self.shape)
        else:
            self.alpha = reshape_list([0 for a in beta],self.shape)
        if self.cooling:
            alphalist=[[alph]*len(self.ag) for alph in alphas]
            self.alpharange = [reshape_list(alphas,self.shape) for alphas in alphalist]
            self.cooling_max = np.max(alphas)
            self.cooling_len = len(alphas)
        if len(beta)>0:
            self.beta = reshape_list(beta,self.shape)
        self.calc_joint()

    def graph(self):
        draw_graph(tp=self.tp,shp=self.shape)

    def set_dims(self,U,M):
        self.dims = []
        self.dims.append(np.shape(U)[0])
        for a in self.conf[self.tp]['mdims']:
            if a == 'M':
                self.dims += M
            else:
                num = self.shape[int(a)]
                if type(num) == int:
                    self.dims.append(num)
                elif type(num) == tuple:
                    for n in num: self.dims.append(n)
        self.dims.append(np.shape(U)[1])

    def propagate_agents(self):
        self.ag = []
        for node in self.nd:
            self.ag += node.ag

    def get_F_loc(self,index):
        I = np.ones(self.dims)  # array of ones to fill up missing indices
        if len(self.tp) == 1:
            C = [[0],[1]]
        elif len(self.tp) == 2:
            C = [[0,1],[0,0],[1,1]]
        c = C[index]
        F_loc =  np.einsum(self.U,[0,self.joint.r[-1]],I,self.joint.r,self.joint.r)
        for i in range(1,len(self.nd)):
            node = self.nd[i]
            S_rel = rel_surprisal(node.post,node.prior)
            S_rel_prior = rel_surprisal_prior(node.prior,node.p0,self.cp)
            F_loc += -c[i-1]*np.einsum(1.0/self.beta[i],node.beta_r,S_rel,node.post.r,I,self.joint.r,self.joint.r)
            if self.cp:
                F_loc += -c[i-1]*np.einsum(1.0/self.alpha[i],node.beta_r,S_rel_prior,node.prior.r,I,self.joint.r,self.joint.r)
        return F_loc

    def update_F(self,index):
        if len(self.tp) == 1 and self.tp[0]==-1: # RD
            order = [0]
        elif len(self.tp) == 1 and self.tp[0]>=0: # 2step cases (ser and par)
            order = [1,0]
        elif len(self.tp) == 2: # 3step cases
            order = [1,2,0]
        r = [self.nd[i].post.r for i in order]
        Z = np.einsum(self.joint.val,self.joint.r,r[index])+1e-50
        F_loc = self.get_F_loc(index)
        self.F_curr = np.einsum(1.0/Z,r[index],self.joint.val,self.joint.r,F_loc,self.joint.r,r[index])

    def calc_joint(self):
        joint = self.pw.val
        rj = [0]
        for node in self.nd:
            rjnew = range(0,node.post.r[-1]+1)
            joint = np.einsum(joint,rj,node.post.val,node.post.r,rjnew)
            rj = rjnew
        self.joint.val = joint

    def calc_pagw(self):
        self.pagw = np.einsum(1.0/self.pw.val,[0],self.joint.val,self.joint.r,[0,self.joint.r[-1]])

    def calc_EU(self):
        self.EU = np.einsum('ik,ik->i',self.pagw,self.U).dot(self.pw.val)

    def calc_avDKLs(self):
        self.DKL = []
        for node in self.nd:
            D = np.einsum(self.joint.val,self.joint.r,node.DKL,node.post.r[:-1],[])
            self.DKL.append(D)

    def calc_FE(self):
        # calc averDKL with betas:
        DKLw = []
        for i in range(0,len(self.nd)):
            node = self.nd[i]
            D = np.einsum(1.0/self.beta[i],node.beta_r,self.joint.val,self.joint.r,node.DKL,node.post.r[:-1],[])
            DKLw.append(D)
            if self.cp:
                Dp = np.einsum(1.0/self.alpha[i],node.beta_r,self.joint.val,self.joint.r,node.DKLpr,node.prior.r[:-1],[])
        # calc FE:
        self.FE = self.EU
        for D in DKLw:
            self.FE += (-1)*D

    def update_agents(self):
        for node in self.nd: node.extract_agents()

    def iterate(self):
        test = 0
        for i in range(0,10000):
            if self.cooling:
                if i<len(self.alpharange):
                    self.alpha = self.alpharange[i]
                elif i==len(self.alpharange):
                    self.alpha = reshape_list([self.cooling_max for a in range(0,len(self.ag))],self.shape)
            for l in range(0,len(self.nd)):
                self.nd[l].process(self.F_curr,self.beta[l],self.alpha[l],self.joint)
                self.calc_joint()
                self.update_F(l)
            if i > 200+len(self.alpharange) and np.linalg.norm(self.nd[-1].post.val-test) < 1e-10:
                break
            test = self.nd[-1].post.val
        self.calc_pagw()
        self.calc_EU()
        for node in self.nd:
            node.calc_DKL()
            if self.cp: node.calc_DKLpr()
        self.calc_avDKLs()
        self.calc_FE()
        self.update_agents()
