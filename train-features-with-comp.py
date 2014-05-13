# -*- coding: utf-8 -*-
__author__ = 'arenduchintala'
from DifferentiableFunction import DifferentiableFunction
import fst, sys, codecs, pdb
from pprint import pprint
from math import exp as expm
from logadd import logadd
import random as rand
import codecs


global f_names, f_weights, f_init_weights, exp_machines, obs_chain, inp_machines, path, out_path, E_machines
f_names = {}
f_weights = {}
f_init_weights = {}
iteration_num = 0


def write_learned_features(theta):
    global iteration_num
    print 'writing learned features...'
    iteration_num += 1
    writer = codecs.open(out_path + '.' + str(iteration_num), 'w', 'utf-8')
    for k in theta:
        feats = f_ids[k][0].encode('utf-8') + '|||' + f_ids[k][1].encode('utf-8')
        # str('|||'.join(f_ids[k]))
        s = str(k) + '\t' + feats + '\t' + str(theta[k])
        #print s
        writer.write(s.decode('utf8') + '\n')
    writer.flush()
    writer.close()
    if iteration_num == 4:
        exit()

def get_feature_id(isym_id, osym_id, f):
    r = f_names.get((f.isyms.find(isym_id), f.osyms.find(osym_id)), None)
    return r


def gradient(theta):
    write_learned_features(theta)
    print 'getting counts...'
    exp_counts = [fst.LogWeight.ZERO] * (len(f_names) + 1)
    obs_counts = [fst.LogWeight.ZERO] * (len(f_names) + 1)
    for idx, (inp_file, E_file, exp_file, o_chain_file) in enumerate(zip(inp_machines, E_machines, exp_machines, obs_chain)):
        sys.stdout.write('%d \r' % idx)
        sys.stdout.flush()
        inp = fst.read(path + inp_file)
        E = fst.read(path + E_file)
        E_wt = apply_weights(E, theta)
        exp_wt = inp.compose(E_wt)
        o_chain = fst.read(path + o_chain_file)
        obs_wt = exp_wt.compose(o_chain)
        exp_wt = apply_weights(exp_wt, theta)
        (e_counts, o_counts) = get_counts_for_machine(exp_wt, obs_wt)
        exp_counts = accumilate_counts(e_counts, exp_counts)
        obs_counts = accumilate_counts(o_counts, obs_counts)

    grad = {}
    for i, o in f_names:
        k = f_names[i, o]
        ok = obs_counts[k]
        ek = exp_counts[k]
        #exp(c)-exp(e)
        s1 = expm(-float(ok))
        s2 = expm(-float(ek))
        grad[k] = s1 - s2
        #print grad[k], '=', s2, '-', s1, i, o
        #pdb.set_trace()
    print '\ngrad computed'
    return grad


def calculate_counts(f):
    alpha = f.shortest_distance()
    beta = f.shortest_distance(True)
    sparse_counts = {}
    for s in f.states:
        for a in s.arcs:
            f_id = get_feature_id(a.ilabel, a.olabel, f)
            alpha_s = alpha[s.stateid]
            arc_weight = a.weight
            beta_t = beta[a.nextstate]
            val = alpha_s.__mul__(arc_weight)
            val = val.__mul__(beta_t)
            sparse_counts[f_id] = sparse_counts.get(f_id, fst.LogWeight.ZERO) + val
    return sparse_counts


def renormalize(f):
    pathsum = f.shortest_distance(True)[0]
    #print 'pathsum', pathsum
    for state in f.states:
        if state.final != fst.LogWeight.ZERO:  # logweight.ZERO ==> logweight(inf)
            state.final = state.final.__div__(pathsum)
    return f


def get_counts_for_machine(exp_wt, obs_wt):
    re_exp = renormalize(exp_wt)
    re_obs = renormalize(obs_wt)
    e_counts = calculate_counts(re_exp)
    o_counts = calculate_counts(re_obs)
    return e_counts, o_counts


def accumilate_counts(sparse_counts, global_counts):
    for k in sparse_counts:
        if k is not None:
            global_counts[k] += sparse_counts[k]
    return global_counts


def apply_weights(machine, weights):
    #machine.write('before.fst', machine.isyms, machine.osyms)
    for s in machine.states:
        for a in s.arcs:
            f_id = get_feature_id(a.ilabel, a.olabel, machine)
            if f_id is not None:
                a.weight = fst.LogWeight(weights[f_id])
    #machine.write('after.fst', machine.isyms, machine.osyms)
    #exit()
    return machine


def get_likelihood(inp_file, E_file, o_chain_file, theta):
    inp = fst.read(path + inp_file)
    E = fst.read(path + E_file)
    o_chain = fst.read(path + o_chain_file)
    E_wt = apply_weights(E, theta)
    exp_wt = inp.compose(E_wt)  #apply_weights(exp, theta)
    #e_wt.write('e_wt.fst', e_wt.isyms, e_wt.osyms)
    exp_wt = renormalize(exp_wt)
    #e_wt.write('e_norm.fst', e_wt.isyms, e_wt.osyms)
    o = exp_wt.compose(o_chain)
    #o.write('obs.after.fst', o.isyms, o.osyms)
    ll = o.shortest_distance(True)[0]
    return float(ll)


def value(theta):
    likelihood = 0.0
    print 'likelihoods'
    for idx, (inp_file, E_file, exp_file, o_chain_file) in enumerate(zip(inp_machines, E_machines, exp_machines, obs_chain)):
        sys.stdout.write('%d \r' % idx)
        sys.stdout.flush()
        likelihood += get_likelihood(inp_file, E_file, o_chain_file, theta)
    print 'll', likelihood
    return likelihood


if __name__ == '__main__':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
    sys.stdin = codecs.getwriter('utf-8')(sys.stdin)
    path = '/Users/arenduchintala/PycharmProjects/alignment-FSTs/coursera-20/train-20-hmm/'
    #path = 'data/toy2/'
    out_path = path + 'learned.features'
    print 'reading data...'
    f_names = dict((tuple(n.split()[1].split('|||')), int(n.split()[0])) for n in codecs.open(path + 'E.names', 'r', 'utf-8').readlines())
    f_ids = dict((int(n.split()[0]), tuple(n.split()[1].split('|||'))) for n in codecs.open(path + 'E.names', 'r', 'utf-8').readlines())
    f_init_weights = dict((int(n.split()[0]), -float(n.split()[1])) for n in codecs.open(path + 'E.init.weights', 'r', 'utf-8').readlines())
    inp_machines, obs_chain, exp_machines = zip(*[tuple(l.split()) for l in codecs.open(path + 'filenames', 'r').readlines()[1:]])
    E_machines = [e.replace('exp', 'E') for e in exp_machines]

    F = DifferentiableFunction(value, gradient)
    (fopt, theta, return_status) = F.minimize(f_init_weights)
    write_learned_features(theta)
