import numpy as np
from BMA_sysconfig import *

try:
    import networkx as nx
    import matplotlib.pyplot as plt
except:
    print("could not import networkx and/or/ mpl")


class Dist(object):

    def __init__(self,val=0,r=[]):
        self.val = val
        self.r = r

    def initialize(self,dims):
        self.val = normalize(np.random.rand(*[dims[i] for i in self.r]))


def normalize(p,n=1):    # normalizes with respect to the last n entries in shape(p)
    eps = 1e-55
    l = len(np.shape(p))
    rp = range(0,l)
    r = range(0,l-n)
    Z = np.einsum(p,rp,r) + eps
    return np.einsum(1.0/Z,r,p,rp,rp) + eps

def rel_surprisal(post,prior):
    eps = 1e-55
    return np.log(np.einsum(post.val,post.r,1.0/(prior.val+eps),prior.r,post.r)+eps)

def rel_surprisal_prior(prior,p0,cp):
    eps = 1e-55
    if cp:
        return np.log(prior.val/(p0.val+eps)+eps)
    else:
        return 0

def get_DKL(post,prior):
    eps = 1e-55
    return ( np.einsum(post.val,post.r,np.log(post.val+eps),post.r,post.r[:-1])
            -np.einsum(post.val,post.r,np.log(prior.val+eps),prior.r,post.r[:-1]) )

def get_simple_DKL(post,prior):
    eps = 1e-55
    if len(np.shape(post)) == 1:
        return np.maximum(0,np.log(post/(prior+eps) + eps).dot(post))

def count(tpl):
    if type(tpl) == tuple:
        c = 1
        for t in tpl: c = c*t
    elif type(tpl) == int:
        c = tpl
    return c

def reshape_list(lst,shp):
    lst_new = []
    curr = 0
    for num in shp:
        tmp = []
        tmp = np.array(lst[curr:curr+count(num)])
        if type(num) == tuple:
            tmp = tmp.reshape(num)
        curr += count(num)
        lst_new.append(tmp)
    return lst_new

def alpharange(start,stop,step):
    a = start
    while a < stop:
        yield a
        a += step

def gen_node_list(tp):
    if len(tp) == 1 and tp[0]>=0:
        lst = ['X','A']
    elif len(tp) == 1 and tp[0]==-1:
        lst = ['A']
    else:
        lst = []
        for i in range(0,len(tp)):
            lst.append('X' + str(i+1))
        lst.append('A')
    return lst

def translate_shape(shp):
    """
    translate shp into [#ag0,#ag1,#ag2]
    """
    shp_temp = []
    for i in range(0,len(shp)):
        if type(shp[i]) == tuple:
            shp_temp.append(shp[i][0]*shp[i][1])
        else:
            shp_temp.append(shp[i])
    return shp_temp

def get_M(tp,M):
    mdims = BMA_sysconfig[tp]['mdims']
    if len(tp) == 2 and len(mdims) == 1:
        if mdims == ['M']:
            return [M,M]
        else:
            return []
    else:
        return [M for el in mdims if el=='M']

def get_shapes(tp,n): #n = total number of agents
    sh = []
    if tp == (-1,):
        sh.append([1])
    elif tp == (0,):
        sh.append([1,1])
    elif tp == (1,):
        sh.append([1,n-1])
    if n > 2:
        if tp == (0,0) or tp == (0,1) or tp == (2,1):
            sh.append([1,1,1])
        elif tp == (0,2) or tp == (0,3) or tp == (0,5) or tp == (2,2) or tp == (2,3):
            sh.append([1,1,n-2])
        elif tp == (0,4):
            lst = find_inds_that_mult_to_n(n-2)
            for t in lst:
                sh.append([1,1,t])
        elif tp == (1,0) or tp == (1,1):
            sh.append([1,n-2,1])
        elif tp == (1,2):
            t = (n-1)/2
            sh.append([1,t,t])
        elif tp == (1,3) or tp == (1,5):
            for i in range(2,n-2):
                sh.append([1,i,n-1-i])
        elif tp == (1,4):
            for j in range(2,n):
                k = (n-1)/j - 1
                if k*j+j+1 <= n and k>1: sh.append([1,j,(j,k)])
        elif tp == (2,4):
            lst = find_inds_that_mult_to_n(n-2)
            for t in lst:
                if t[0]<=t[1]:
                    sh.append([1,1,t])
    return sh

def get_types():
    return [(-1,),(0,),(1,),
            (0,0),(0,1),(0,2),(0,3),(0,4),(0,5),
            (1,0),(1,1),(1,2),(1,3),(1,4),(1,5),
            (2,1),(2,2),(2,3),(2,4)]

def get_edges(tp):
    if tp == (-1,):
        edges =[('W','A')]
        grey_edges = []
    elif tp == (0,):
        edges =[('W','X'), ('X','A')]
        grey_edges = []
    elif tp == (1,):
        edges =[('W','X'), ('W','A')]
        grey_edges = [('X','A')]
    elif tp == (0,0):
        edges =[('W','X1'), ('X1','X2'), ('X2','A')]
        grey_edges = []
    elif tp == (0,1):
        edges = [('W','X1'), ('X1','X2'), ('X2','A'), ('X1','A')]
        grey_edges = []
    elif tp == (0,2):
        edges = [('W','X1'), ('X1','X2'), ('X2','A'), ('X1','A')]
        grey_edges = [('X1','A')]
    elif tp == (0,3):
        edges = [('W','X1'), ('X1','X2'), ('X2','A'), ('X1','A')]
        grey_edges = [('X2','A')]
    elif tp == (0,4):
        edges = [('W','X1'), ('X1','X2'), ('X2','A'), ('X1','A'), ('W','A')]
        grey_edges = [('X1','A'),('X2','A')]
    elif tp == (0,5):
        edges = [('W','X1'), ('X1','X2'), ('X2','A'), ('W','A')]
        grey_edges = [('X2','A')]
    elif tp == (1,0):
        edges = [('W','X1'), ('X1','X2'), ('W','X2'), ('X2','A')]
        grey_edges = [('X1','X2')]
    elif tp == (1,1):
        edges = [('W','X1'), ('X1','X2'), ('W','X2'), ('X2','A'), ('X1','A')]
        grey_edges = [('X1','X2')]
    elif tp == (1,2):
        edges = [('W','X1'), ('X1','X2'), ('W','X2'), ('X2','A'), ('X1','A')]
        grey_edges = [('X1','X2'), ('X1','A')]
    elif tp == (1,3):
        edges = [('W','X1'), ('X1','X2'), ('W','X2'), ('X2','A'), ('X1','A')]
        grey_edges = [('X1','X2'), ('X2','A')]
    elif tp == (1,4):
        edges = [('W','X1'), ('X1','X2'), ('W','X2'), ('X2','A'), ('X1','A'), ('W','A')]
        grey_edges = [('X1','X2'), ('X1','A'), ('X2','A')]
    elif tp == (1,5):
        edges = [('W','X1'), ('X1','X2'), ('W','X2'), ('X2','A'), ('W','A')]
        grey_edges = [('X1','X2'), ('X2','A')]
    elif tp == (2,1):
        edges = [('W','X1'), ('X1','A'), ('W','X2'), ('X2','A')]
        grey_edges = []
    elif tp == (2,2):
        edges = [('W','X1'), ('X1','A'), ('W','X2'), ('X2','A')]
        grey_edges = [('X1','A')]
    elif tp == (2,3):
        edges = [('W','X1'), ('X1','A'), ('W','X2'), ('X2','A')]
        grey_edges = [('X2','A')]
    elif tp == (2,4):
        edges = [('W','X1'), ('X1','A'), ('W','X2'), ('X2','A'), ('W','A')]
        grey_edges = [('X2','A'),('X1','A')]
    return edges, grey_edges

def gen_graph_entities(tp,ag=[0,0,0]):
    G = nx.DiGraph()
    edges, grey_edges = get_edges(tp)
    G.add_edges_from(edges)
    black_edges = [edge for edge in G.edges() if edge not in grey_edges]
    values = [0.2 for node in G.nodes()]
    if len(tp) == 2:
        positions = nx.spring_layout(G,pos={'W':(0.0,0.0),'X1':(.5,.2),'A':(1.25,0.0),'X2':(.75,-0.2)},fixed=['W','X1','X2','A'])
        if ag[0] > 0:
            labels={'W':'$W$','X1':'{}'.format(ag[0]),'X2':'{}'.format(ag[1]),'A':'{}'.format(ag[2]),}
        else:
            labels={'W':'$W$','X1':'$X_1$','X2':'$X_2$','A':'$A$',}
    elif len(tp) == 1:
        if tp[0]==-1:
            positions = nx.spring_layout(G,pos={'W':(0.0,0.0),'A':(1.25,0.0)},fixed=['W','A'])
            if ag[0] > 0:
                labels={'W':'$W$','A':'{}'.format(1),}
            else:
                labels={'W':'$W$','A':'$A$',}
        else:
            positions = nx.spring_layout(G,pos={'W':(0.0,0.0),'X':(.625,.2),'A':(1.25,0.0)},fixed=['W','X','A'])
            if ag[0] > 0:
                labels={'W':'$W$','X':'{}'.format(ag[0]),'A':'{}'.format(ag[1]),}
            else:
                labels={'W':'$W$','X':'$X$','A':'$A$',}
    return G,black_edges,grey_edges,values,positions,labels

def draw_graph(tp,shp=[1,1,1]):
    G,black_edges,grey_edges,values,pos,labels = gen_graph_entities(tp,ag=translate_shape(shp))
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('Greys'),node_color = values, node_size = 2000)
    nx.draw_networkx_edges(G, pos, edgelist=grey_edges, style='dashed', arrows=True)
    nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=True)
    nx.draw_networkx_labels(G, pos, labels, font_size=25)
    plt.axis('off')
    plt.show()

def draw_graph_onax(tp,shp,ax,fontsize=15):
    ag = translate_shape(shp)
    G,black_edges,grey_edges,values,pos,labels = gen_graph_entities(tp,ag)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('Greys'),node_color = values, node_size = 1000,ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=grey_edges, style='dashed', arrows=True,ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=True, ax=ax)
    nx.draw_networkx_labels(G, pos, labels, font_size=fontsize,ax=ax)
    ax.axis('off')

def draw_graphs(tps,fsize=(15,13)):

    f1 = int(np.sqrt(len(tps)))
    f2 = int(np.ceil(float(len(tps))/f1))

    fig, ax = plt.subplots(f1,f2,figsize=fsize)
    for i in range(0,len(tps)):
        tp = tps[i]
        G,black_edges,grey_edges,values,pos,labels = gen_graph_entities(tp)
        if f1>1 and f2>1:
            j,k = np.unravel_index(i,(f1,f2))
            axx = ax[j,k]
        else:
            axx = ax[i]
        nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('Greys'),node_color = values, node_size = 600,ax=axx)
        nx.draw_networkx_edges(G, pos, edgelist=grey_edges, style='dashed', arrows=True, ax=axx)
        nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=True, ax=axx)
        nx.draw_networkx_labels(G, pos, labels, font_size=15, ax=axx)
        axx.set_title(str(tps[i]),loc='left',verticalalignment='top')
    for i in range(0,f1*f2):
        if f1>1 and f2>1:
            j,k = np.unravel_index(i,(f1,f2))
            axx = ax[j,k]
        else:
            axx = ax[i]
        axx.axis('off')
    #plt.tight_layout()
    plt.show()
