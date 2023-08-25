#############
## PREPARE ##
#############

#workspace
from scipy.stats import binom
import graph_tool.all as gt
import leidenalg as la
import networkx as nx
import operator as oe
import NetBone as nb
import igraph as ig
import pandas as pd
import numpy as np
import random
import string

#functions
def noise_corrected(data, approximation = True):
    g = nx.from_pandas_edgelist(data, edge_attr = "weight", create_using = nx.Graph())
    n = sum(nx.get_edge_attributes(g, name = "weight").values())
    for i, j, w in g.edges(data = "weight"):
        ni= g.degree(i, weight = "weight")
        nj= g.degree(j, weight = "weight")
        mean_prior_probability = ((ni * nj) / n) * (1 / n)
        kappa = n / (ni * nj)
        if approximation:
            g[i][j]["p_value"] = 1 - binom.cdf(w, n, mean_prior_probability)
        else:
            score = ((kappa * w) - 1) / ((kappa * w) + 1)
            var_prior_probability = (1 / (n ** 2)) * (ni * nj * (n - ni) * (n - nj)) / ((n ** 2) * ((n - 1)))
            alpha_prior = (((mean_prior_probability ** 2) / var_prior_probability) * (1 - mean_prior_probability)) - mean_prior_probability
            beta_prior = (mean_prior_probability / var_prior_probability) * (1 - (mean_prior_probability ** 2)) - (1 - mean_prior_probability)
            alpha_post = alpha_prior + w
            beta_post = n - w + beta_prior
            expected_pij = alpha_post / (alpha_post + beta_post)
            variance_nij = expected_pij * (1 - expected_pij) * n
            d = (1.0 / (ni * nj)) - (n * ((ni + nj) / ((ni * nj) ** 2)))
            variance_cij = variance_nij * (((2 * (kappa + (w * d))) / (((kappa * w) + 1) ** 2)) ** 2)
            sdev_cij = variance_cij ** 0.5
            g[i][j]["nc_sdev"] = sdev_cij
            g[i][j]["score"] = score
    return nb.Backbone(g, name = "Noise Corrected Filter", column = "p_value", ascending = True)

#maximum community number
def maxcomn(p, m = 2, w = False):
    indices, sizes = np.unique(p.membership, return_counts = True)
    while len(indices) > m:
        tm = indices[np.argmin(sizes)]
        tm_s = sizes[np.argmin(sizes)]
        ap = p.aggregate_partition()
        if w:
            am = pd.DataFrame(ap.graph.get_adjacency(attribute = "weight"))
        else:
            am = pd.DataFrame(ap.graph.get_adjacency(attribute = None))
        am = am.loc[indices, indices]
        div = [tm_s] * len(sizes)
        aw = am.loc[:, tm]
        aw = aw / div
        aw = aw.drop(tm) #chooses one or more based on highest relative density
        nw = aw.index[np.argwhere(aw.values == np.max(aw)).flatten()]
        N = []
        for n in nw:
            N.append(sizes[np.argwhere(indices == n).flatten()][0])
        nw = nw[np.argmin(N)] #merges to smallest highest-relative-density community [by index if more than one]
        ap.move_node(tm, nw)
        p.from_coarse_partition(ap)
        indices, sizes = np.unique(p.membership, return_counts = True)
    return p

#minimum community size
def mincoms(p, m = 2, w = False):
    indices, sizes = np.unique(p.membership, return_counts = True)
    tmall = []
    tmall_s = []
    for i in np.arange(len(sizes)):
        if sizes[i] < m:
            tmall.append(indices[i])
            tmall_s.append(sizes[i])
    while len(tmall) > 0:
        ap = p.aggregate_partition()
        if w:
            am = pd.DataFrame(ap.graph.get_adjacency(attribute = "weight"))
        else:
            am = pd.DataFrame(ap.graph.get_adjacency(attribute = None))
        am = am.loc[indices, indices]
        tm = tmall[np.argmin(tmall_s)]
        tm_s = tmall_s[np.argmin(tmall_s)]
        div = [tm_s] * len(sizes)
        aw = am.loc[:, tm]
        aw = aw / div
        aw = aw.drop(tm) #chooses one based on highest relative density
        nw = aw.index[np.argwhere(aw.values == np.max(aw)).flatten()]
        N = []
        for n in nw:
            N.append(sizes[np.argwhere(indices == n).flatten()][0])
        nw = nw[np.argmin(N)] #merges to smallest highest-relative-density community [by index if more than one]
        ap.move_node(tm, nw)
        p.from_coarse_partition(ap)
        indices, sizes = np.unique(p.membership, return_counts = True)
        tmall = []
        tmall_s = []
        for i in np.arange(len(sizes)):
            if sizes[i] < m:
                tmall.append(indices[i])
                tmall_s.append(sizes[i])
    return p

def exin(elist, vlist, weights = False, adaptive = False, entire = False):
    """External-Internal Index by Krackhardt and Stern (1988) based on pairwise comparisons between blocks. Takes two Pandas dataframes as input. The 'elist' needs to contain all edges with 'source' and 'target' columns; if 'weights' is True, an additional column 'weight' must be present. The 'vlist' needs to contain all vertices with 'vertex' and 'member' columns. If 'adaptive' is True, the index is normalised by the product of the block sizes. If 'entire' is False, a block matrix with block sizes along the diagonal is returned. If 'entire' is True, the weighted average of the pairwise comparisons is returned with weights corresponding to the product of the block sizes. Requires Numpy 'as np' and Pandas 'as pd'."""

    el = elist
    nl = vlist
    nl = nl.rename(columns = {"vertex": "source", "member": "member_s"})
    el = pd.merge(el, nl, on = "source", how = "left")
    nl = nl.rename(columns = {"source": "target", "member_s": "member_t"})
    el = pd.merge(el, nl, on = "target", how = "left")
    nl = nl.rename(columns = {"target": "vertex", "member_t": "member"})
    el["internal"] = np.where(el["member_s"] == el["member_t"], 1, 0)
    el["external"] = np.where(el["member_s"] != el["member_t"], 1, 0)
    
    if weights:
        el["internal_w"] = np.where(el["internal"] == 1, el["weight"], 0)
        el["external_w"] = np.where(el["external"] == 1, el["weight"], 0)
        el["internal"] = el["internal_w"]
        el["external"] = el["external_w"]
    
    bname, bsize = np.unique(nl["member"], return_counts = True)
    bs = pd.DataFrame(np.zeros(shape = (len(bsize), len(bsize))))
    for j in np.flip(np.arange(len(bname))):
        bs = bs.rename(columns = {j: bname[j]}, index = {j: bname[j]})
    np.fill_diagonal(bs.values, bsize)
    
    bm = bs.copy()
    np.fill_diagonal(bm.values, 0)
    for edge in range(len(el)):
        s = el["member_s"][edge]
        t = el["member_t"][edge]
        if s == t:
            bm.loc[s, t] += el["internal"][edge]
        else:
            bm.loc[s, t] += el["external"][edge]
    bm = bm.add(bm.transpose())
    np.fill_diagonal(bm.values, np.diag(bm) / 2)

    a = bname.tolist() * len(bname)
    b = np.repeat(bname, len(bname)).tolist()

    if adaptive:
        bn_internal = bs.copy()
        np.fill_diagonal(bn_internal.values, 0)
        for j in bname:
            bn_internal.loc[j, j] = (bm.loc[j, j]) / (bs.loc[j, j] * (bs.loc[j, j] - 1) * 0.5)

        bs_divider = bs.copy()
        np.fill_diagonal(bs_divider.values, 0)
        for k in a:
            for l in b:
                bs_divider.loc[k, l] = (bs.loc[k, k] * (bs.loc[l, l] - 1) * 0.5)

        bn = bm.div(bs_divider)
        np.fill_diagonal(bn.values, np.diag(bn_internal))
    else:
        bn = bm.copy()
    
    ei = bs.copy()
    np.fill_diagonal(ei.values, 0)
    for k in a:
        for l in b:
            lefthand = -(bn.loc[k, k] + bn.loc[l, l] - bn.loc[k, l] - bn.loc[l, k])
            righthand = (bn.loc[k, k] + bn.loc[l, l] + bn.loc[k, l] + bn.loc[l, k])
            if righthand != 0:
                ei.loc[k, l] = lefthand / righthand
            else:
                ei.loc[k, l] = 0
    np.fill_diagonal(ei.values, np.diag(bs))

    if entire:
        bs_product = bs.copy()
        np.fill_diagonal(bs_product.values, 0)
        for k in a:
            for l in b:
                bs_product.loc[k, l] = bs.loc[k, k] * bs.loc[l, l]
        a_values = ei.values[np.triu_indices_from(ei.values, k = 1)]
        a_weights = bs_product.values[np.triu_indices_from(bs_product.values, k = 1)]
        if sum(a_weights) == 0:
            ei = 0
        else:
            ei = np.average(a_values, weights = a_weights)
    return ei

def estimate_th_in_th_out(A, S, T):
    #theta_in and theta_out assuming degree-corrected model
    num_in = 0; denom_in = 0
    num_out = 0; denom_out = 0
    for t in np.arange(T):
        twom = np.count_nonzero(A[t])
        deg = np.sum(A[t], 0)
        deg_r = [0] * np.max(S[t])
        twom_in = 0
        
        #sum of degrees in each group, and number of within-community edges
        for r in np.arange(1, 1 + np.max(S[t])):
            #r = 1
            idx = S[t] == r
            deg_r[r - 1] = np.sum(deg[idx.index[idx]])
            twom_in = twom_in + np.count_nonzero(A[t].loc[idx.index[idx], idx.index[idx]])

        num_in = num_in + twom_in
        num_out = num_out + twom - twom_in
        term = np.sum(np.power(deg_r, 2)) / twom
        denom_in = denom_in + term
        denom_out = denom_out + twom - term

    th_in = num_in / denom_in
    if denom_out != 0:
        th_out = num_out / denom_out
    else:
        th_out = 0
    return th_in, th_out
    
def estimate_K(S):
    K = np.unique(S)
    if 0 in K:
        K = len(K) - 1
    else:
        K = len(K)
    return K
    
def persistence(S, T, categorical = False):
    fers = pd.DataFrame(np.zeros(shape = (T, T), dtype = "int"))
    pers = pd.DataFrame(np.zeros(shape = (T, T), dtype = "int"))
    for k in pers.index:
        for l in pers.columns:
            m = S.iloc[:, [k, l]].replace(0, np.nan).dropna()
            fers.iloc[k, l] = m.shape[0] #number of common vertices
            pers.iloc[k, l] = np.sum(m.iloc[:, 0] == m.iloc[:, 1])
    np.fill_diagonal(fers.values, 0)
    np.fill_diagonal(pers.values, 0)
    if categorical:
        pers = np.sum(np.triu(pers)) / np.sum(np.triu(fers))
    else:
        k = np.arange(1, T)
        l = np.arange(T - 1)
        fsum = []
        psum = []
        for o in np.arange(T - 1):
            fsum.append(fers.loc[k[o], l[o]])
            psum.append(pers.loc[k[o], l[o]])
        pers = np.sum(psum) / np.sum(fsum)
    return pers

def estimate_p(S, K, T, coupling_type = "ordinal"):    
    if coupling_type == "ordinal":
        pers = persistence(S, T, categorical = False)
        if pers == 1:
            p = 1
        elif pers == 0:
            p = 0
        elif K == 1:
            p = 1
        else:
            p = np.max([(K * pers - 1) / (K - 1), 0])
    if coupling_type == "categorical_uniform":
        pers = persistence(S, T, categorical = True)
        if pers == 1:
            p = 1
        elif pers == 0:
            p = 0
        elif K == 1:
            p = 1
        else:
            coeff = 2 * (1 - 1 / K) / (T * (T - 1))
            p0 = 0.5
            def f(pp):
                a = np.arange(1, T)
                b = np.flip(a)
                c = []
                for t in np.arange(T - 1):
                    c.append(a[t] * pp ** b[t])
                prob = coeff * np.sum(c) + 1 / K
                return prob - pers
            from scipy.optimize import least_squares
            p = least_squares(f, p0, bounds = (0, np.nextafter(1, 0))).x
    return float(p)

def estimate_SBM_parameters(A, S, T, coupling_type = "ordinal", uniform = True, compute_p = True):
    if (coupling_type == "ordinal") & (uniform == True):
        th_in = estimate_th_in_th_out(A = A, S = S, T = T)[0]
        th_out = estimate_th_in_th_out(A = A, S = S, T = T)[1]
        K = estimate_K(S = S)
        mb = S.replace(0, np.nan).dropna().astype("int")
        p = estimate_p(S = mb, K = K, T = T, coupling_type = "ordinal")
    
    if (coupling_type == "ordinal") & (uniform == False):
        th_in = []
        th_out = []
        K = []
        for t in np.arange(T):
            nw = A[t]
            mb = S[t].replace(0, np.nan).dropna().astype("int")
            th_in.append(estimate_th_in_th_out(A = [nw], S = [mb], T = 1)[0])
            th_out.append(estimate_th_in_th_out(A = [nw], S = [mb], T = 1)[1])
            K.append(estimate_K(S = mb))
        p = []
        for t in np.arange(T - 1):
            mb = S.loc[:, [t, t + 1]].replace(0, np.nan).dropna().astype("int")
            p.append(estimate_p(S = mb, K = K[t + 1], T = 2, coupling_type = "ordinal"))

    if (coupling_type == "categorical") & (uniform == True):
        th_in = estimate_th_in_th_out(A = A, S = S, T = T)[0]
        th_out = estimate_th_in_th_out(A = A, S = S, T = T)[1]
        K = estimate_K(S = S)
        mb = S.replace(0, np.nan).dropna().astype("int")
        p = estimate_p(S = mb, K = K, T = T, coupling_type = "categorical_uniform")
        
    if (coupling_type == "categorical") & (uniform == False):
        th_in = []
        th_out = []
        K = []
        for t in np.arange(T):
            nw = A[t]
            mb = S[t].replace(0, np.nan).dropna().astype("int")
            th_in.append(estimate_th_in_th_out(A = [nw], S = [mb], T = 1)[0])
            th_out.append(estimate_th_in_th_out(A = [nw], S = [mb], T = 1)[1])
            K.append(estimate_K(S = mb))
        if compute_p == True:
            from itertools import permutations
            num_permutations = 0
            p = pd.DataFrame(np.zeros(shape = (T, T)))
            for prm in permutations(np.arange(T)):
                #prm = (1, 2, 4, 3, 0, 5)
                Sp = S.loc[:, prm]
                Sp.columns = np.arange(T)
                Kp = []
                for t in np.arange(T):
                    mb = Sp[t].replace(0, np.nan).dropna().astype("int")
                    Kp.append(estimate_K(S = mb))
                p_t = []
                for t in np.arange(T - 1):
                    mb = Sp.loc[:, [t, t + 1]].replace(0, np.nan).dropna().astype("int")
                    p_t.append(estimate_p(S = mb, K = Kp[t + 1], T = 2, coupling_type = "ordinal"))
                
                p_m = pd.DataFrame(np.zeros(shape = (T, T)))
                for s in np.arange(T):
                    for r in np.arange(T):
                        if s == r - 1:
                            p_m.iloc[s, r] = p_t[s]
                        elif s == r:
                            p_m.iloc[s, r] = 0 #0 ("We assume that pss = 0", p. 10), 1 / T (pss = 0.83 if only 1 community), or T (pss = 1 if only 1 community?
                p_m = p_m.loc[np.argsort(prm), np.argsort(prm)]
                p_m.index = np.arange(T)
                p_m.columns = np.arange(T)
                p += p_m
                num_permutations += 1
            p = p / num_permutations * T #1 / T (pss = 0.83 if only 1 community), or T (pss = 1 if only 1 community?
        else:
            p = pd.DataFrame(np.zeros(shape = (T, T)))
            
    return th_in, th_out, K, p

def loglik_intra(A, S, T, th_in, th_out): #Equation 10
    ls = 0
    for t in np.arange(T):
        #t = 1
        s = S[t].replace(0, np.nan).dropna().astype("int").reset_index(drop = True)
        g = ig.Graph.Adjacency(A[t].values.tolist(), mode = "undirected")
        g.vs["name"] = A[t].index.values.tolist()
        td = []
        for v in g.vs:
            if v.degree() == 0:
                td.append(v)
        g.delete_vertices(td)
        if th_out[t] != 0:
            gamma = (th_in[t] - th_out[t]) / (np.log(th_in[t]) - np.log(th_out[t]))
            lg = np.log(th_in[t]) - np.log(th_out[t])
        else:
            th_out_x = th_in[t] / 10 #np.nextafter(0, 1) #OR np.nextafter(1, 0)?
            th_in_x = th_in[t] - th_out_x
            gamma = (th_in_x - th_out_x) / (np.log(th_in_x) - np.log(th_out_x))
            lg = np.log(th_in_x) - np.log(th_out_x)
        a = pd.DataFrame(g.get_adjacency(attribute = None))
        a.index = g.vs["name"]
        a.columns = g.vs["name"]
        ss = 0
        for k in np.arange(g.vcount()):
            for l in np.arange(g.vcount()):
                if k != l:
                    aa = a.iloc[k, l] - gamma * ((g.vs[k].degree() * g.vs[l].degree()) / (2 * g.ecount()))
                    if s[k] == s[l]:
                        delta = 1
                    else:
                        delta = 0
                    ss += (aa * delta)
        ls += lg * ss
    return ls

def loglik_inter(S, p, K, T): #Equation 28
    from itertools import permutations
    ss = 0
    for n in np.arange(len(S)):
        #n = 15
        s = 0
        num_permutations = 0
        num_m = 0
        for prm in permutations(np.arange(T)):
            #prm = (4, 3, 5, 0, 1, 2)
            m = 1
            num_m += 1
            for t in np.arange(1, T):
                #t = 1
                if (S.iloc[n, prm[t - 1]] != 0) & (S.iloc[n, prm[t]] != 0):
                    num_permutations += 1 / (T - 1) #moved num_permutations after the check that n appears in both layers
                    pprm = p.iloc[prm[t - 1], prm[t]]
                    if pprm == 1:
                        pprm = np.nextafter(1, 0)
                    Kprm = K[prm[t]]
                    if S.iloc[n, prm[t - 1]] == S.iloc[n, prm[t]]:
                        delta = 1
                    else:
                        delta = 0
                    m = m * (1 - pprm) * (1 + pprm / (1 - pprm) * Kprm * delta)
            s += m
        num_permutations = int(np.round(num_permutations, 0))
        if s == num_m: #otherwise s will be 720 due to m being assigned before the check that n appears in both layers
            s = s / num_m
        else:
            s = s / num_permutations
        #print(s)
        #print(num_m)
        #print(num_permutations)
        ss += np.log(s)
    return ss

def loglik(gs, cg, rs, cs, ls, rn, mcn, mcs, qf, op):
    
    #rs = [0.70] * cg.vcount()
    #cs = [0.05] * cg.ecount()
    #ls = [1.00] * (cg.vcount() + 1)
    #rn = 50
    #mcn = 4
    #mcs = 6
        
    #consensus partition
    cg.vs["slice"] = gs
    cg.es["weight"] = cs
    lrs, inter, fg = la.slices_to_layers(cg, slice_attr = "slice", vertex_id_attr = "name", weight_attr = "weight")
    pt = []
    for seed in np.arange(rn):
        part = []
        for layer in np.arange(len(lrs)):
            part.append(qf(lrs[layer], weights = None, resolution_parameter = rs[layer]))
        part_inter = la.CPMVertexPartition(inter, weights = "weight", resolution_parameter = 0)
        op.set_rng_seed(seed)
        op.optimise_partition_multiplex(part + [part_inter], n_iterations = -1, layer_weights = ls)
        pt.append(part[0].membership)
    for e in fg.es:
        if e["type"] == "interslice":
            e["layer"] = -1
    fg = fg.to_graph_tool(vertex_attributes = {"name": "string", "slice": "string"}, edge_attributes = {"weight": "float", "layer": "int"})
    sd = 1; gt.seed_rng(sd); np.random.seed(sd)
    pm = gt.PartitionModeState(bs = pt, relabel = True, converge = True)
    pv = pm.get_marginal(fg)
    pb = fg.new_vp("int")
    for v in fg.vertices():
        pb[v] = np.argmax(pv[v])
    
    #constrain minimum community number and maximum community size
    cn = []
    pc = pd.DataFrame({"pb": pb, "layer": fg.vp.slice, "vertex": fg.vp.name})
    pc["layer"] = pc["layer"].astype("int")
    for layer in pc["layer"].unique():
        #layer = 4
        td = []
        for v in lrs[layer].vs:
            if v["slice"] != layer:
                td.append(v)
        lrs[layer].delete_vertices(td)
        if pc[pc["layer"] == layer]["vertex"].tolist() == lrs[layer].vs["name"] == False:
            print("!!!")
        p = la.RBConfigurationVertexPartition(lrs[layer], initial_membership = None, weights = None, resolution_parameter = 0)
        #print(np.max(pc[pc["layer"] == layer]["pb"]), len(pc[pc["layer"] == layer]["pb"]))
        p.set_membership(pc[pc["layer"] == layer]["pb"])
        #print(np.unique(p.membership, return_counts = True))
        p = maxcomn(p, m = mcn, w = False)
        p = mincoms(p, m = mcs, w = False)
        #print(np.unique(p.membership, return_counts = True))
        cn = cn + p.membership
    pc["pb"] = cn

    #relabel communities by size
    vc = pc["pb"].value_counts()
    mp = {v:k for k, v in dict(enumerate(vc.index)).items()}
    pc["pb"] = pc["pb"].map(mp)
    for v in np.arange(fg.num_vertices()):
        pb[v] = pc["pb"][v]
    
    #store community structure
    fg.vp.pb = pb
    
    #input for planted partition model
    D = pd.DataFrame({"vertex": fg.vp.name, "layer": fg.vp.slice, "member": pb})
    A = []
    for g in gs:
        pa = pd.DataFrame(np.zeros((len(D["vertex"].unique()), len(D["vertex"].unique())), dtype = "int"))
        pa.index = D["vertex"].unique()
        pa.columns = D["vertex"].unique()
        a = pd.DataFrame(g.get_adjacency(attribute = None))
        a.index = g.vs["name"]
        a.columns = g.vs["name"]
        for c in a.columns:
            if c in pa.columns:
                pa.loc[:, c] = a.loc[:, c]
        A.append(pa.fillna(0).astype("int"))
    T = len(A)
    S = pd.DataFrame({"vertex": D["vertex"].unique()})
    for l in D["layer"].unique():
        L = D[D["layer"] == l].drop("layer", axis = 1)
        L.columns = ["vertex", l]
        S = pd.merge(S, L, on = "vertex", how = "left")
    S = S.drop("vertex", axis = 1).fillna(- 1).astype("int") + 1
    S.index = D["vertex"].unique().tolist()
    S.columns = np.arange(T)

    #pad isolates
    #S = S.replace(0, np.nan)
    #v, c = np.unique(S, return_counts = True)
    #if np.isnan(v[- 1]):
    #    fill = len(v)
    #    for column in S.columns:
    #        for row in S.index:
    #            if np.isnan(S.loc[row, column]):
    #                S.loc[row, column] = fill
    #                fill += 1
    #S = S.astype("int")

    #loglik for planted partition model
    if np.mean(cs) > 0:
        th_in, th_out, K, p = estimate_SBM_parameters(A = A, S = S, T = T, coupling_type = "categorical", uniform = False, compute_p = True)
        ll_intra = loglik_intra(A = A, S = S, T = T, th_in = th_in, th_out = th_out)
        ll_inter = loglik_inter(S = S, p = p, K = K, T = T)
    else:
        th_in, th_out, K, p = estimate_SBM_parameters(A = A, S = S, T = T, coupling_type = "categorical", uniform = False, compute_p = False)
        ll_intra = loglik_intra(A = A, S = S, T = T, th_in = th_in, th_out = th_out)
        ll_inter = 0.0
    ll = ll_intra + ll_inter
    
    return ll, ll_intra, ll_inter, K, fg

def fitppm(gs, cg, rs_min = 0.60, rs_max = 1.40, rs_stp = 0.01, cs_min = 0.00, cs_max = 0.20, cs_stp = 0.01, ls_min = 1.00, ls_max = 1.00, ls_stp = 0.00, rn = 20, mcn = 1000, mcs = 1, mi = 20):
    """
    Iterates the Leiden algoritm by Traag et al. (2019) with different sets of layer-specific resolution, layer-pair-specific coupling, and layer-specific layer weight parameters and computes the log-likehood for each to find the partition from which the multilayer network is most likely to emerge. Determines initial resolution and coupling parameters through grid search using every 4th element of sequence [ _min, _max, _stp ]. Returns the optimal parameter values, the log-likelihood, and the full graph with the partition as an internal vertex attribute. Requires numpy, pandas, igraph, leidenalg, and graph-tool.
    
    ds: multilayer network as pandas.DataFrame edgelist. First columns must specify "source" and "target".
    gs: list of igraph objects as layers.
    cg: coupling graph that specifies interlayer edges.
    rs: minimum, maximum, and step for resolution parameters.
    cs: minimum, maximum, and step for coupling parameters.
    ls: minimum, maximum, and step for coupling parameters [if step == 0, each layer has the same weight]. 
    rn: number of runs of the Leiden algorithm to optimise modularity for each set of parameters [consensus clustering].
    mcn: maximum number of communities per layer.
    mcs: maximum community size per layer.
    mi: maximum number of iterations.
    """
    
    #rs_min = 0.70
    #rs_max = 1.00
    #rs_stp = 0.02
    #cs_min = 0.005
    #cs_max = 0.105
    #cs_stp = 0.010
    #ls_min = 1.00
    #ls_max = 1.00
    #ls_stp = 0.00
    #rn = 20
    #mcn = 4
    #mcs = 6
    #mi = 20
    
    #sanity check
    if len(gs) < 2:
        exit("gs contains less than two igraph objects, exiting...")
    if (rs_min < 0) or (rs_max < 0) or (rs_min > rs_max) or (rs_stp < 0) or ((rs_min == rs_max) and (rs_stp > 0)):
        from sys import exit
        exit("invalid resolution max, min, and/or stp, exiting...")
    if (cs_min < 0) or (cs_max < 0) or (cs_min > cs_max) or (cs_stp < 0) or ((cs_min == cs_max) and (cs_stp > 0)):
        from sys import exit
        exit("invalid coupling max, min, and/or stp, exiting...")
    if (ls_min < 0) or (ls_max < 0) or (ls_min > ls_max) or (ls_stp < 0) or ((ls_min == ls_max) and (ls_stp > 0)):
        from sys import exit
        exit("invalid layer weight max, min, and/or stp, exiting...")
    
    #input
    print("---------------------------------")
    print("[γ] resolution                  :", "[min]", "{:1.3f}".format(rs_min), "[max]", "{:1.3f}".format(rs_max), "[stp]", "{:1.3f}".format(rs_stp))
    print("[ω] coupling                    :", "[min]", "{:1.3f}".format(cs_min), "[max]", "{:1.3f}".format(cs_max), "[stp]", "{:1.3f}".format(cs_stp))
    print("[β] layer weight                :", "[min]", "{:1.3f}".format(ls_min), "[max]", "{:1.3f}".format(ls_max), "[stp]", "{:1.3f}".format(ls_stp))
    print("modularity runs                 :", rn)
    print("maximum community number        :", mcn)
    print("minimum community size          :", mcs)    
    print("maximum multilayer iterations   :", mi)
    print("---------------------------------")
    
    #settings
    op = la.Optimiser()
    qf = la.RBConfigurationVertexPartition
    
    #grid resolution
    if (rs_stp == 0) and (rs_min == rs_max):
        rs = [rs_min]
    else:
        rs = np.round(np.arange(rs_min, rs_max, rs_stp), 10)
    
    #grid coupling              
    if (cs_stp == 0) and (cs_min == 0) and (cs_max == 0):
        cs = [0.0]
        gd = pd.DataFrame(np.zeros(shape = (len(rs), len(cs))))
        gd.index = rs
        gd.columns = cs
        gl = gd.shape
        gd = gd.iloc[::4, :]
    elif (cs_stp == 0) and (cs_min != 0) and (cs_max != 0) and (cs_min == cs_max):
        cs = [cs_min]
        gd = pd.DataFrame(np.zeros(shape = (len(rs), len(cs))))
        gd.index = rs
        gd.columns = cs
        gl = gd.shape
        gd = gd.iloc[::4, :]
    else:
        cs = np.round(np.arange(cs_min, cs_max, cs_stp), 10)
        gd = pd.DataFrame(np.zeros(shape = (len(rs), len(cs))))
        gd.index = rs
        gd.columns = cs
        gl = gd.shape
        gd = gd.iloc[::4, ::4]
        
    gd = gd.rename_axis(index = "γ", columns = "ω")
    print("grid size                       :", gd.shape[0] * gd.shape[1])
    print("---------------------------------")
    ls = [1.0] * cg.vcount() + [1.0]
    for r in gd.index:
        for c in gd.columns:
            rgrid = [r] * cg.vcount()
            cgrid = [c] * cg.ecount()
            ll, ll_intra, ll_inter, K, fg = loglik(gs = gs, cg = cg, rs = rgrid, cs = cgrid, ls = ls, rn = rn, mcn = mcn, mcs = mcs, qf = qf, op = op)
            if (1 in K):
                gd.loc[r, c] = -np.inf
            else:
                gd.loc[r, c] = ll
    print(np.round(gd, 1))
    print("---------------------------------")
    
    rs = [gd.max(axis = 1).idxmax()] * cg.vcount()
    cs = [gd.max(axis = 0).idxmax()] * cg.ecount()
    ll, ll_intra, ll_inter, K, fg = loglik(gs = gs, cg = cg, rs = rs, cs = cs, ls = ls, rn = rn, mcn = mcn, mcs = mcs, qf = qf, op = op)    
    print("[γ] initial                     :", "{:1.3f}".format(gd.max(axis = 1).idxmax()))
    print("[ω] initial                     :", "{:1.3f}".format(gd.max(axis = 0).idxmax()))
    print("[β] initial                     :", "{:1.3f}".format(ls[0]))
    print("[l] initial intra               :", np.round(ll_intra, 1))
    print("[l] initial inter               :", np.round(ll_inter, 1))
    print("[l] initial total               :", np.round(ll, 1))
    print("[K] initial                     :", K)
    print("---------------------------------")
    
    #refine
    bl = ll
    for iteration in np.arange(mi):
        
        #counter
        change = 0
        
        #resolution
        if gl[0] != 1:
            for ix in np.arange(len(rs)):
                curre = rs[ix]
                param = curre
                for i in np.round(np.arange(rs_min, rs_max, rs_stp), 10):
                    rs[ix] = i
                    ll, ll_intra, ll_inter, K, fg = loglik(gs = gs, cg = cg, rs = rs, cs = cs, ls = ls, rn = rn, mcn = mcn, mcs = mcs, qf = qf, op = op)
                    if (1 in K):
                        ll = -np.inf
                    if ll > bl:
                        bl = ll
                        param = i
                        print("improvement                     :", np.round(bl, 1))
                if curre != param:
                    change += 1
                rs[ix] = param

        #coupling
        if gl[1] != 1:
            for ix in np.arange(len(cs)):
                curre = cs[ix]
                param = curre
                for i in np.round(np.arange(cs_min, cs_max, cs_stp), 10):
                    cs[ix] = i
                    ll, ll_intra, ll_inter, K, fg = loglik(gs = gs, cg = cg, rs = rs, cs = cs, ls = ls, rn = rn, mcn = mcn, mcs = mcs, qf = qf, op = op)
                    if (1 in K):
                        ll = -np.inf
                    if ll > bl:
                        bl = ll
                        param = i
                        print("improvement                     :", np.round(bl, 1))
                if curre != param:
                    change += 1
                cs[ix] = param
        
        if ls_stp != 0:

            #layer weight
            for ix in np.arange(len(ls)):
                curre = ls[ix]
                param = curre
                for i in np.round(np.arange(ls_min, ls_max, ls_stp), 10):
                    ls[ix] = i
                    ll, ll_intra, ll_inter, K, fg = loglik(gs = gs, cg = cg, rs = rs, cs = cs, ls = ls, rn = rn, mcn = mcn, mcs = mcs, qf = qf, op = op)
                    if (1 in K):
                        ll = -np.inf
                    if ll > bl:
                        bl = ll
                        param = i
                        print("improvement                     :", np.round(bl, 1))
                if curre != param:
                    change += 1
                ls[ix] = param
        
        if change == 0:
            
            ll, ll_intra, ll_inter, K, fg = loglik(gs = gs, cg = cg, rs = rs, cs = cs, ls = ls, rn = rn, mcn = mcn, mcs = mcs, qf = qf, op = op)
            print("required iterations             :", iteration + 1, "/", mi, "[max]")
            print("---------------------------------")
            print("[γ] optimal                     :", ["{:1.3f}".format(e) for e in rs])
            print("[ω] optimal                     :", ["{:1.3f}".format(e) for e in cs])
            print("[β] optimal                     :", ["{:1.3f}".format(e) for e in ls])
            print("[l] optimal intra               :", np.round(ll_intra, 1))
            print("[l] optimal inter               :", np.round(ll_inter, 1))
            print("[l] optimal total               :", np.round(ll, 1))
            print("[K] optimal                     :", K)
            print("---------------------------------")
            return rs, cs, ls, ll, fg
            break
        
        else:
            print("---------------------------------")
            print("iteration                       :", iteration + 1, "/", mi, "[max]")
            print("parameters updated              :", change, "/", len(rs) + len(cs) + (len(ls) - 1), "[max]")
            print("---------------------------------")

###########
## BONES ##
###########

#data
d = pd.read_csv("/m/triton/scratch/work/malkama5/complexCoalition/networkData/fs.txt", delimiter = ";", header = None, names = ["source", "target", "weight", "layer"], dtype = {"weight": "float", "layer": "int"})

#backbone
dn = []
gs = []
bb = pd.DataFrame()
for l in d["layer"].unique():
    tb = d[d["layer"] == l]
    vc = ig.Graph.DataFrame(tb, directed = False, use_vids = False).vcount()
    dn.append(ig.Graph.DataFrame(tb, directed = False, use_vids = False).density())
    nc = noise_corrected(data = tb, approximation = True)
    df = nx.to_pandas_edgelist(nb.Filters.threshold_filter(nc, 0.05))
    df["layer"] = l
    vc = vc - ig.Graph.DataFrame(df, directed = False, use_vids = False).vcount()
    if vc != 0:
        if vc == 1:
            print("Layer", l, "has", vc, "vertex less than the original graph.")
        else:
            print("Layer", l, "has", vc, "vertices less than the original graph.")
    gs.append(ig.Graph.DataFrame(df, directed = False, use_vids = False))
    bb = pd.concat([bb, df], axis = 0, ignore_index = True)

#density
for l in bb["layer"].unique():
    print(np.round(dn[l], 3), "layer:", l, "original weighted")
    print(np.round(ig.Graph.DataFrame(bb[bb["layer"] == l], directed = False, use_vids = False).density(), 3), "layer:", l, "backbone binary")

#store
hdr = bb.columns.tolist()
hdr = ";".join([str(e) for e in hdr])
np.savetxt("/m/triton/scratch/work/malkama5/complexCoalition/networkData/bb.txt", bb.values, delimiter = ";", fmt = "%s", header = hdr, comments = "")

####################
## COUPLING GRAPH ##
####################

#coupling graph
cg = ig.Graph()
cg.add_vertices(len(gs))
cg.vs["id"] = range(len(gs))
cg.add_edges([
    #temporal
    (0, 1),
    (2, 3),
    (4, 5),
    #modal
    (0, 2), (0, 4),
    (2, 4),
    (1, 3), (1, 5),
    (3, 5),
    #temporal-modal
    (0, 3), (0, 5),
    (2, 1), (2, 5),
    (4, 1), (4, 3)])

############
## DETECT ##
############

print("monolayer")
rs, cs, ls, ll, fg = fitppm(gs = gs, cg = cg, rs_min = 0.30, rs_max = 1.20, rs_stp = 0.01, cs_min = 0.000, cs_max = 0.000, cs_stp = 0.000, rn = 50, mcn = 3, mcs = 2)

print("multilayer")
rs, cs, ls, ll, fg = fitppm(gs = gs, cg = cg, rs_min = 0.30, rs_max = 1.20, rs_stp = 0.01, cs_min = 0.001, cs_max = 0.161, cs_stp = 0.005, rn = 50, mcn = 3, mcs = 2)

###########
## STORE ##
###########

fg.save("/m/triton/scratch/work/malkama5/complexCoalition/networkResult/ppm_" + str(int(np.round(ll))) + ".xml")

##############
## CONTINUE ##
##############

#load
ll = 2681
fg = gt.load_graph("/m/triton/scratch/work/malkama5/complexCoalition/networkResult/ppm_" + str(ll) + ".xml")

#collapse
src = []
trg = []
wgt = []
for e in fg.edges():
    if fg.ep.layer[e] >= 0:
        s, t = e
        for v in fg.vertices():
            if fg.vertex_index[v] == s:
                src.append(fg.vp.name[v])
            if fg.vertex_index[v] == t:
                trg.append(fg.vp.name[v])
        wgt.append(fg.ep.weight[e])
clp = pd.DataFrame({"source": src, "target": trg, "weight": wgt})
clp = ig.Graph.DataFrame(clp, directed = False, use_vids = False).to_graph_tool(vertex_attributes = {"name": "string"}, edge_attributes = {"weight": "float"})

#################
## COORDINATES ##
#################

#partition
sd = 1; gt.seed_rng(sd); np.random.seed(sd)
st = gt.minimize_blockmodel_dl(clp, state = gt.PPBlockState, multilevel_mcmc_args = dict(B_min = 2, B_max = 3))
sd = 1; gt.seed_rng(sd); np.random.seed(sd)
lo = gt.sfdp_layout(clp, groups = st.b, gamma = 0.001, kappa = 20)

#fix
st.draw(pos = lo, vertex_text = clp.vp.name, vertex_font_size = 10)
for v in clp.vertices():
    if clp.vp.name[v] == "FI044":
        lo[v][0] = lo[v][0] + 0.0
        lo[v][1] = lo[v][1] + 1.0
    if clp.vp.name[v] == "FI134":
        lo[v][0] = lo[v][0] - 0.5
        lo[v][1] = lo[v][1] - 0.0
    if clp.vp.name[v] == "FI019":
        lo[v][0] = lo[v][0] - 0.5
        lo[v][1] = lo[v][1] - 0.0
    if clp.vp.name[v] == "FI138":
        lo[v][0] = lo[v][0] - 0.5
        lo[v][1] = lo[v][1] - 0.0
    if clp.vp.name[v] == "FI129":
        lo[v][0] = lo[v][0] - 0.5
        lo[v][1] = lo[v][1] - 0.0
    if clp.vp.name[v] == "FI046":
        lo[v][0] = lo[v][0] - 0.5
        lo[v][1] = lo[v][1] - 0.0
    if clp.vp.name[v] == "FI131":
        lo[v][0] = lo[v][0] - 0.5
        lo[v][1] = lo[v][1] - 0.0
st.draw(pos = lo, vertex_text = clp.vp.name, vertex_font_size = 4)

#layerwise
cr = pd.DataFrame({"vertex": fg.vp.name, "layer": fg.vp.slice})
cr["layer"] = cr["layer"].astype("int")
x = []
y = []
for v in clp.vertices():
    x.append(lo[v][0] * 1.0) #widen
    y.append(lo[v][1])
cr = pd.merge(cr, pd.DataFrame({"vertex": clp.vp.name, "x": x, "y": y}), on = "vertex", how = "left")
ef = 1.0 #expansion
cr["x"] = np.where(cr["layer"] == 0, cr["x"] + 0 * ef, cr["x"])
cr["x"] = np.where(cr["layer"] == 1, cr["x"] + 14 * ef, cr["x"])
cr["x"] = np.where(cr["layer"] == 2, cr["x"] + 0 * ef, cr["x"])
cr["x"] = np.where(cr["layer"] == 3, cr["x"] + 14 * ef, cr["x"])
cr["x"] = np.where(cr["layer"] == 4, cr["x"] + 0 * ef, cr["x"])
cr["x"] = np.where(cr["layer"] == 5, cr["x"] + 14 * ef, cr["x"])

cr["y"] = np.where(cr["layer"] == 0, cr["y"] + 0 * ef, cr["y"])
cr["y"] = np.where(cr["layer"] == 1, cr["y"] + 0 * ef, cr["y"])
cr["y"] = np.where(cr["layer"] == 2, cr["y"] + 13 * ef, cr["y"])
cr["y"] = np.where(cr["layer"] == 3, cr["y"] + 13 * ef, cr["y"])
cr["y"] = np.where(cr["layer"] == 4, cr["y"] + 26 * ef, cr["y"])
cr["y"] = np.where(cr["layer"] == 5, cr["y"] + 26 * ef, cr["y"])

cr["y"] = np.where(cr["layer"] == 0, cr["y"] + 0 * ef, cr["y"])
cr["y"] = np.where(cr["layer"] == 1, cr["y"] + 6 * ef, cr["y"])
cr["y"] = np.where(cr["layer"] == 2, cr["y"] + 0 * ef, cr["y"])
cr["y"] = np.where(cr["layer"] == 3, cr["y"] + 6 * ef, cr["y"])
cr["y"] = np.where(cr["layer"] == 4, cr["y"] + 0 * ef, cr["y"])
cr["y"] = np.where(cr["layer"] == 5, cr["y"] + 6 * ef, cr["y"])

cr["x"] = np.where(cr["layer"] == 0, cr["x"] + 0 * ef, cr["x"])
cr["x"] = np.where(cr["layer"] == 1, cr["x"] + 0 * ef, cr["x"])
cr["x"] = np.where(cr["layer"] == 2, cr["x"] - 3 * ef, cr["x"])
cr["x"] = np.where(cr["layer"] == 3, cr["x"] - 3 * ef, cr["x"])
cr["x"] = np.where(cr["layer"] == 4, cr["x"] + 0 * ef, cr["x"])
cr["x"] = np.where(cr["layer"] == 5, cr["x"] + 0 * ef, cr["x"])

lo = fg.new_vp("vector<double>")
for v in np.arange(fg.num_vertices()):
    lo[v] = [cr["x"][v], cr["y"][v]]

############
### PLOT ###
############

#vertex size
vz = pd.DataFrame()
for l in gs:
    dg = l.degree(mode = "all", loops = False)
    dg = (dg - np.min(dg)) / (np.max(dg) - np.min(dg))    
    vz = pd.concat([vz, pd.DataFrame({"vertex": l.vs["name"], "layer": l.es["layer"][0], "vz": dg})], axis = 0, ignore_index = True)
cr = pd.merge(cr, vz, left_on = ["vertex", "layer"], right_on = ["vertex", "layer"], how = "left")
vz = fg.new_vp("float")
for v in np.arange(fg.num_vertices()):
    vz[v] = cr["vz"][v]

#vertex colour
vc = fg.new_vp("string")
cpie = ["#008e97", "#fc4c02", "#C204af", "#49d5de", "#d9774e", "#C771be"]
#cpie = ["#cb7c77ff", "#68d359ff", "#6b42c8ff", "#c9d73dff", "#c555cbff", "#aed688ff", "#502e71ff", "#c49a3fff", "#6a7dc9ff", "#d7652dff", "#7cd5c8ff", "#c5383cff", "#507d41ff", "#cf4c8bff", "#5d8d9cff", "#722e41ff", "#c8b693ff", "#33333cff", "#c6a5ccff", "#674c2aff"]
for v in fg.vertices():
    for c in np.arange(len(np.unique(fg.vp.pb.a))):
        if fg.vp.pb[v] == c:
            vc[v] = cpie[c]

#edge width, order, linetype
ew = fg.new_ep("float")
eo = fg.new_ep("int")
et = fg.new_ep("vector<float>")
for e in fg.edges():
    if fg.ep.layer[e] < 0:
        ew[e] = 0.1
        eo[e] = 0
        et[e] = [0, 0.2, 0, 0.2]
    else:
        ew[e] = 0.2
        eo[e] = 1
        et[e] = []

#plot
gt.BlockState(fg).draw(pos = lo, vertex_fill_color = vc, vertex_color = "#ffffff", vertex_size = gt.prop_to_size(vz, 2, 4, log = False), vorder = vz, vertex_aspect = 5, edge_pen_width = ew, eorder = eo, edge_dash_style = et, output = "/m/triton/scratch/work/malkama5/complexCoalition/networkPlot/comCoaFig_x_ppm_" + str(ll) + ".svg")

########################
## MUTUAL INFORMATION ##
########################

dl = pd.DataFrame({"vertex": fg.vp.name, "layer": fg.vp.slice, "pb": fg.vp.pb})
hd = ";".join([str(e) for e in dl.columns.tolist()])
np.savetxt("/m/triton/scratch/work/malkama5/complexCoalition/networkData/dl_" + str(ll) + ".txt", dl.values, delimiter = ";", fmt = "%s", header = hd, comments = "")

dw = pd.DataFrame({"vertex": np.unique(dl["vertex"])})
for l in dl["layer"].unique():
    x = dl[dl["layer"] == l]
    x = x.drop("layer", axis = 1)
    x = x.rename(columns = {"pb": str(l)})
    dw = pd.merge(dw, x, on = "vertex", how = "left")

mi = pd.DataFrame(np.zeros(shape = (len(dl["layer"].unique()), len(dl["layer"].unique()))))
for r in np.arange(len(mi.index)):
    for c in np.arange(len(mi.columns)):
        m = dw.drop("vertex", axis = 1).iloc[:, [r, c]].dropna().astype("int")
        m.columns = [0, 1]
        mi.loc[r, c] = np.round(gt.reduced_mutual_information(m[m.columns[0]], m[m.columns[1]], norm = True), 2)
np.savetxt("/m/triton/scratch/work/malkama5/complexCoalition/networkResult/mi_" + str(ll) + ".txt", mi.values, delimiter = ";", fmt = "%s")

###############
## SCHEMATIC ##
###############

qfn = la.RBConfigurationVertexPartition
opt = la.Optimiser()
gc = []
random.seed(0)
gc.append(ig.Graph.SBM(26, pref_matrix = [[0.5, 0.1], [0.1, 0.5]], block_sizes = [11, 15]))
random.seed(0)
gc.append(ig.Graph.SBM(26, pref_matrix = [[0.6, 0.2], [0.2, 0.6]], block_sizes = [11, 15]))
random.seed(0)
gc.append(ig.Graph.SBM(26, pref_matrix = [[0.6, 0.2], [0.2, 0.6]], block_sizes = [11, 15]))
random.seed(0)
gc.append(ig.Graph.SBM(26, pref_matrix = [[0.7, 0.1], [0.1, 0.7]], block_sizes = [11, 15]))
random.seed(0)
gc.append(ig.Graph.Erdos_Renyi(n = gc[0].vcount(), m = gc[0].ecount(), directed = False, loops = False))
random.seed(0)
gc.append(ig.Graph.SBM(26, pref_matrix = [[0.6, 0.2], [0.2, 0.6]], block_sizes = [11, 15]))
random.seed(0)
gc.append(ig.Graph.SBM(26, pref_matrix = [[0.6, 0.2], [0.2, 0.6]], block_sizes = [18, 8]))
random.seed(0)
gc.append(ig.Graph.SBM(26, pref_matrix = [[0.6, 0.2], [0.2, 0.6]], block_sizes = [8, 18]))
for l in np.arange(len(gc)):
    i = []
    for j in np.arange(26):
        i.append(string.ascii_uppercase[j])
    gc[l].vs["id"] = np.arange(26).tolist()
    gc[l].vs["name"] = i
    gc[l].es["layer"] = l

np.random.seed(1)
gc[2].delete_vertices(np.random.randint(0, 25, 6).tolist())
np.random.seed(2)
gc[3].delete_vertices(np.random.randint(3, 22, 7).tolist())
for l in range(len(gc)):
    gc[l].es["weight"] = 1
    normaliser = 1 / float(sum(gc[l].es["weight"])) * 10
    gc[l].es["weight"] = [x * normaliser for x in gc[l].es["weight"]]

cg = ig.Graph()
cg.add_vertices(len(gc))
cg.vs["id"] = range(len(gc))
cg.add_edges([
(0, 1), (0, 2), (0, 3),
(1, 2), (1, 3),
(2, 3), (2, 4), (2, 5),
(3, 4), (3, 5),
(4, 5), (4, 6), (4, 7),
(5, 6), (5, 7),
(6, 7)])
ig.plot(cg, layout = "circle", vertex_label = cg.vs["id"])
cg.es["weight"] = 0.025
cg.vs["slice"] = gc
lrs, inter, fugr = la.slices_to_layers(cg, slice_attr = "slice", vertex_id_attr = "name", weight_attr = "weight")
part = [
qfn(lrs[0], weights = "weight", resolution_parameter = 1.0),
qfn(lrs[1], weights = "weight", resolution_parameter = 1.0),
qfn(lrs[2], weights = "weight", resolution_parameter = 1.0),
qfn(lrs[3], weights = "weight", resolution_parameter = 1.0),
qfn(lrs[4], weights = "weight", resolution_parameter = 0.0),
qfn(lrs[5], weights = "weight", resolution_parameter = 1.0),
qfn(lrs[6], weights = "weight", resolution_parameter = 1.0),
qfn(lrs[7], weights = "weight", resolution_parameter = 1.0)]
part_inter = la.CPMVertexPartition(inter, node_sizes = "node_size", weights = "weight", resolution_parameter = 0)
opt.set_rng_seed(0)
diff = opt.optimise_partition_multiplex(part + [part_inter], n_iterations = - 1, layer_weights = None)
fugr.vs["memb"] = part[0].membership
                              
el = fugr.get_edge_dataframe()
vl = fugr.get_vertex_dataframe()
el["source"] = el["source"].replace(vl["name"])
el["target"] = el["target"].replace(vl["name"])
el = el[el["layer"] == 0]
el["layer"] = el["layer"].astype("int")

sg = ig.Graph.DataFrame(el, directed = False, use_vids = False).to_graph_tool(vertex_attributes = {"name": "string"}, edge_attributes = {"layer": "int", "weight": "float", "type": "string"})
gt.seed_rng(1); np.random.seed(1)
pos = gt.sfdp_layout(sg, max_iter = 0)
gt.graph_draw(sg, pos = pos)
crd = pd.DataFrame({"actor": sg.vp.name, "x": 0, "y": 0})
for cd in range(len(crd)):
    crd.iloc[cd, 1] = pos[cd][0]
    crd.iloc[cd, 2] = pos[cd][1]

x = pd.DataFrame({"actor": fugr.vs["name"], "layer": fugr.vs["slice"]})
x["actor"] = x["actor"] + "_" + x["layer"].astype("str")
fugr.vs["name"] = x["actor"].tolist()

el = fugr.get_edge_dataframe()
vl = fugr.get_vertex_dataframe()
el["source"] = el["source"].replace(vl["name"])
el["target"] = el["target"].replace(vl["name"])
el = el.loc[:, ["source", "target", "type"]]
sg = ig.Graph.DataFrame(el, directed = False, use_vids = False).to_graph_tool(vertex_attributes = {"name": "string"}, edge_attributes = {"type": "string"})
x = pd.DataFrame({"actor": sg.vp.name})
z = pd.DataFrame({"actor": fugr.vs["name"], "memb": fugr.vs["memb"]})
x = pd.merge(x, z, on ="actor", how = "left")
x["layer"] = x["actor"].str[2:].astype("int")
x["actor"] = x["actor"].str[:1]
x = pd.merge(x, crd, on ="actor", how = "left")

x["memb"] = np.where((x["layer"] == 6) & (x["memb"] == 0), 2, x["memb"])
x["memb"] = np.where((x["layer"] == 6) & (x["memb"] == 1), 0, x["memb"])
x["memb"] = np.where((x["layer"] == 6) & (x["memb"] == 2), 1, x["memb"])

exf = 1.1
x["x"] = np.where(x["layer"] == 0, x["x"] + 0 * exf, x["x"])
x["x"] = np.where(x["layer"] == 1, x["x"] + 0 * exf, x["x"])
x["x"] = np.where(x["layer"] == 2, x["x"] + 10 * exf, x["x"])
x["x"] = np.where(x["layer"] == 3, x["x"] + 10 * exf, x["x"])
x["x"] = np.where(x["layer"] == 4, x["x"] + 20 * exf, x["x"])
x["x"] = np.where(x["layer"] == 5, x["x"] + 20 * exf, x["x"])
x["x"] = np.where(x["layer"] == 6, x["x"] + 30 * exf, x["x"])
x["x"] = np.where(x["layer"] == 7, x["x"] + 30 * exf, x["x"])

x["y"] = np.where(x["layer"] == 0, x["y"] + 0 * exf, x["y"])
x["y"] = np.where(x["layer"] == 1, x["y"] + 10 * exf, x["y"])
x["y"] = np.where(x["layer"] == 2, x["y"] + 0 * exf, x["y"])
x["y"] = np.where(x["layer"] == 3, x["y"] + 10 * exf, x["y"])
x["y"] = np.where(x["layer"] == 4, x["y"] + 0 * exf, x["y"])
x["y"] = np.where(x["layer"] == 5, x["y"] + 10 * exf, x["y"])
x["y"] = np.where(x["layer"] == 6, x["y"] + 0 * exf, x["y"])
x["y"] = np.where(x["layer"] == 7, x["y"] + 10 * exf, x["y"])

x["y"] = np.where(x["layer"] == 0, x["y"] + 0 * exf, x["y"])
x["y"] = np.where(x["layer"] == 1, x["y"] + 0 * exf, x["y"])
x["y"] = np.where(x["layer"] == 2, x["y"] + 5 * exf, x["y"])
x["y"] = np.where(x["layer"] == 3, x["y"] + 5 * exf, x["y"])
x["y"] = np.where(x["layer"] == 4, x["y"] + 10 * exf, x["y"])
x["y"] = np.where(x["layer"] == 5, x["y"] + 10 * exf, x["y"])
x["y"] = np.where(x["layer"] == 6, x["y"] + 15 * exf, x["y"])
x["y"] = np.where(x["layer"] == 7, x["y"] + 15 * exf, x["y"])

cpie = ["#008e97", "#fc4c02", "#C204af", "#49d5de", "#d9774e", "#C771be"]
vo = sg.new_vp("int")
co = sg.new_vp("string")
sh = sg.new_vp("string")
b = sg.new_vp("int")
ps = sg.new_vp("vector<double>")
for v in range (sg.num_vertices()):
    b[v] = x["memb"][v]
    ps[v] = [x["x"][v], x["y"][v]]

ew = sg.new_ep("float")
eo = sg.new_ep("int")
lt = sg.new_ep("vector<float>")
for e in sg.edges():
    if sg.ep.type[e] == "interslice":
        ew[e] = 0.15
        eo[e] = 0
        lt[e] = [0, 0.2, 0, 0.2]
    else:
        ew[e] = 0.3
        eo[e] = 1
        lt[e] = []
    
u = x["memb"].unique()
for v in sg.vertices():
    if b[v] == u[0]:
        co[v] = cpie[0]
    else:
        co[v] = cpie[1]
    vo[v] = 1
    sg.vp.name[v] = ""
    sh[v] = "circle"
        
ss = gt.BlockState(sg, b = b)
ss.draw(pos = ps, vertex_size = 3, vertex_fill_color = co, vertex_color = "#00000000", vertex_aspect = 5, edge_pen_width = ew, edge_dash_style = lt, eorder = eo, output = "/m/triton/scratch/work/malkama5/complexCoalition/networkPlot/comCoaFig_x_schematic.svg")

###################
### POWER/POLAR ###
###################

#power attribution
i0 = pd.read_csv("/m/triton/scratch/work/malkama5/complexCoalition/networkData/repinf2014.txt", delimiter = ";", header = None, names = ["vertex", "inf"], dtype = {"vertex": "str", "inf": "float"})
i0["inf"] -= i0["inf"].min()
i0["inf"] /= i0["inf"].max()
i1 = pd.read_csv("/m/triton/scratch/work/malkama5/complexCoalition/networkData/repinf2020.txt", delimiter = ";", header = None, names = ["vertex", "inf"], dtype = {"vertex": "str", "inf": "float"})
i1["inf"] -= i1["inf"].min()
i1["inf"] /= i1["inf"].max()

#prepare
dl = pd.DataFrame({"vertex": fg.vp.name, "layer": fg.vp.slice, "pb": fg.vp.pb})
dl["layer"] = dl["layer"].astype("int")
vlists = []
for l in dl["layer"].unique():
    vl = dl[dl["layer"] == l]
    if (l == 0) or (l == 2) or (l == 4):
        vl = pd.merge(vl, i0, on = "vertex", how = "left")
    if (l == 1) or (l == 3) or (l == 5):
        vl = pd.merge(vl, i1, on = "vertex", how = "left")
    vl = vl.rename(columns = {"pb": "member"})
    vl = vl.drop("layer", axis = 1)
    vlists.append(vl)
elists = []
for l in np.arange(len(bb["layer"].unique())):
    el = bb[bb["layer"] == l]
    el = el.rename(columns = {"from": "source", "to": "target"})
    elists.append(el)

#power/polar matrices
popo = []
for l in dl["layer"].unique():
    ei = np.round(exin(elist = elists[l], vlist = vlists[l], weights = False, adaptive = True, entire = False) * (-1), 2)
    
    cs = np.diag(ei) * (-1)
    if (l == 0) or (l == 2) or (l == 4):
        cs = np.round((cs / 96), 2)
    if (l == 1) or (l == 3) or (l == 5):
        cs = np.round((cs / 103), 2)
    
    ri = vlists[l].groupby("member")["inf"].sum()
    #for v in np.arange(len(ri)):
    for v in ri.index:
        if (l == 0) or (l == 2) or (l == 4):
            ri[v] = np.round((ri[v] / i0["inf"].sum()), 2)
        if (l == 1) or (l == 3) or (l == 5):
            ri[v] = np.round((ri[v] / i1["inf"].sum()), 2)
    
    np.fill_diagonal(ei.values, np.nan)
    ei["rel_size"] = cs
    ei["rel_power"] = ri
    popo.append(ei)

#settings
cpie = ["#008e97", "#fc4c02", "#C204af", "#49d5de", "#d9774e", "#C771be"]
random.seed(0)
#cd = ig.Graph.Full(n = len(dl["pb"].unique()), directed = False).layout_circle().coords
cd = ig.Graph.Full(n = len(dl["pb"].unique()), directed = False).layout_grid().coords
x = []
y = []
for c in cd:
    x.append(c[0])
    y.append(c[1])

#square
popof = pd.DataFrame()
for l, p in enumerate(popo):
    pp = pd.DataFrame(np.zeros(shape = (len(dl["pb"].unique()), len(dl["pb"].unique())))).replace(0, np.nan)
    pp.index = dl["pb"].unique()
    pp.columns = dl["pb"].unique()
    pp["rel_size"] = np.nan
    pp["rel_power"] = np.nan
    pp["x"] = x
    pp["y"] = y
    cp = [cpie[:len(dl["pb"].unique()-1)][e] for e in dl["pb"].unique().tolist()]
    pp["colo"] = cp
    pp["laye"] = l
    for c in p.columns:
        pp.loc[:, c] = p.loc[:, c]
    popof = pd.concat([popof, pp], axis = 0, ignore_index = True)

#fill
popof["rel_size"] = popof["rel_size"].fillna(0)
popof["rel_power"] = popof["rel_power"].fillna(0)

#round
popof["rel_size"] = popof["rel_size"] * 100
popof["rel_power"] = popof["rel_power"] * 100

#coordinates
ef = 0.15
popof["x"] = np.where(popof["laye"] == 0, popof["x"] + 0 * ef, popof["x"])
popof["x"] = np.where(popof["laye"] == 1, popof["x"] + 15 * ef, popof["x"])
popof["x"] = np.where(popof["laye"] == 2, popof["x"] + 0 * ef, popof["x"])
popof["x"] = np.where(popof["laye"] == 3, popof["x"] + 15 * ef, popof["x"])
popof["x"] = np.where(popof["laye"] == 4, popof["x"] + 0 * ef, popof["x"])
popof["x"] = np.where(popof["laye"] == 5, popof["x"] + 15 * ef, popof["x"])

popof["y"] = np.where(popof["laye"] == 0, popof["y"] + 0 * ef, popof["y"])
popof["y"] = np.where(popof["laye"] == 1, popof["y"] + 0 * ef, popof["y"])
popof["y"] = np.where(popof["laye"] == 2, popof["y"] + 12 * ef, popof["y"])
popof["y"] = np.where(popof["laye"] == 3, popof["y"] + 12 * ef, popof["y"])
popof["y"] = np.where(popof["laye"] == 4, popof["y"] + 24 * ef, popof["y"])
popof["y"] = np.where(popof["laye"] == 5, popof["y"] + 24 * ef, popof["y"])

#create
elist = pd.DataFrame()
for l in popof["laye"].unique():
    p = popof[popof["laye"] == l]
    a = p.loc[:, dl["pb"].unique()].fillna(0)
    np.fill_diagonal(a.values, 1)
    a = ig.Graph.Weighted_Adjacency(a, mode = "upper")
    el = a.get_edge_dataframe()
    vl = a.get_vertex_dataframe()
    el["source"] = el["source"].replace(vl["name"])
    el["target"] = el["target"].replace(vl["name"])
    elist = pd.concat([elist, el], axis = 0, ignore_index = True)
g = ig.Graph.DataFrame(elist, directed = False, use_vids = False).to_graph_tool(vertex_attributes = {"name": "string"}, edge_attributes = {"weight": "float"})

#edge attributes
et = g.new_ep("string")
for e in g.edges():
    et[e] = "{:0.2f}".format(g.ep.weight[e])
g.ep.et = et
gt.remove_self_loops(g)
g.ep.feight = g.new_ep("float")
g.ep.feight.a = np.flip(g.ep.weight.a)
ctrl = g.new_ep("vector<double>")
for e in g.edges():
    ctrl[e] = [
        0.00, 0.00, #initial point
        0.63, 0.00, #segment 1 control point 1
        0.63, 0.00, #segment 1 control point 2
        1.00, 0.00] #segment 1 endpoint

#vertex attributes
ps = g.new_vp("vector<double>")
co = g.new_vp("string")
rs = g.new_vp("float")
rp = g.new_vp("float")
rt = g.new_vp("string")
for v in np.arange(g.num_vertices()):
    ps[v] = [popof["x"][v], popof["y"][v]]
    co[v] = popof["colo"][v]
    rs[v] = popof["rel_size"][v]
    rp[v] = popof["rel_power"][v]
    rt[v] = "{:0.0f}".format(popof["rel_size"][v]) + "|" + "{:0.0f}".format(popof["rel_power"][v])
g.vp.ps = ps
g.vp.co = co
g.vp.rs = rs
g.vp.rp = rp
g.vp.rt = rt
td = []
for v in g.vertices():
    if v.out_degree() == 0:
        td.append(v)
for v in reversed(sorted(td)):
    g.remove_vertex(v)

#plot
gt.BlockState(g).draw(pos = g.vp.ps, vertex_fill_color = g.vp.co, vertex_color = "#00000000", vertex_size = gt.prop_to_size(g.vp.rs, 40, 80), vertex_aspect = 1, vertex_text = g.vp.rt, vertex_font_size = 12, vertex_text_position = -1, vertex_text_color = "#000000", vertex_text_offset = [0, 0], edge_pen_width = gt.prop_to_size(g.ep.weight, 1, 10), edge_text = g.ep.et, edge_text_color = "#000000", edge_font_size = 12, edge_text_distance = 8, edge_text_parallel = True, edge_control_points = None, eorder = g.ep.feight, output = "/m/triton/scratch/work/malkama5/complexCoalition/networkPlot/comCoaFig_x_power_" + str(ll) + ".svg")
