# -*- coding: utf-8 -*-
__author__ = 'arenduchintala'
from DifferentiableFunction import DifferentiableFunction
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import approx_fprime
from scipy.optimize import check_grad
import fst, sys, codecs, pdb
from pprint import pprint
from math import exp as expm
from math import sqrt
from logadd import logadd_of_list as logadd_list
import codecs
import optparse
from align import natural_sort as ns

global f_names, f_weights, f_init_weights, obs_machines, path, out_path
f_names = {}
f_weights = {}
f_init_weights = {}
iteration_num = 0


def write_learned_features(theta):
    global iteration_num
    print 'writing learned features...'
    iteration_num += 1
    writer = codecs.open(out_path + '.' + str(iteration_num), 'w', 'utf-8')
    for idx, k in enumerate(theta):
        feats = f_ids[idx][0].encode('utf-8') + '|||' + f_ids[idx][1].encode('utf-8')
        # str('|||'.join(f_ids[k]))
        s = str(idx) + '\t' + feats + '\t' + str(k)
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
    pdb.set_trace()
    write_learned_features(theta)
    print 'getting counts...'
    exp_counts = [fst.LogWeight.ZERO] * (len(f_names) + 1)
    obs_counts = [fst.LogWeight.ZERO] * (len(f_names) + 1)
    for idx, (obs_trellis_file) in enumerate(obs_machines):
        sys.stdout.write('%d \r' % idx)
        sys.stdout.flush()
        obs = fst.read(path + obs_trellis_file)
        #obs_c = fst.read(path + obs_chain_file)
        obs_trellis = apply_weights(obs, theta)
        obs_trellis = renormalize(obs_trellis)
        e_counts, o_counts = get_counts(obs_trellis)
        exp_counts = accumilate_counts(e_counts, exp_counts)
        obs_counts = accumilate_counts(o_counts, obs_counts)

    grad = np.zeros(len(theta))
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


def renormalize(trellis):
    '''
    make outgoing arcs for each state sum to 1. (0 in logweight)
    '''
    for state in trellis.states:
        arc_weights = []
        for arc in state.arcs:
            if arc.ilabel != 0 and arc.olabel != 0:
                arc_weights.append(float(arc.weight))
        total_outgoing_prob = logadd_list(arc_weights)
        for arc in state.arcs:
            if arc.ilabel != 0 and arc.olabel != 0:
                arc.weight = arc.weight.__div__(fst.LogWeight(total_outgoing_prob))
    return trellis


def get_counts(trellis):
    alpha = trellis.shortest_distance()
    beta = trellis.shortest_distance(True)
    sparse_expected_counts = {}
    sparse_observed_counts = {}
    for s in trellis.states:
        for a in s.arcs:
            f_id = get_feature_id(a.ilabel, a.olabel, trellis)
            alpha_s = alpha[s.stateid]
            beta_s = beta[s.stateid]
            beta_t = beta[a.nextstate]
            '''
            compute expected feature counts
            val = alpha_s * beta_s * arc_weight
            '''
            val_exp = alpha_s.__mul__(beta_s).__mul__(a.weight)
            sparse_expected_counts[f_id] = sparse_expected_counts.get(f_id, fst.LogWeight.ZERO) + val_exp
            '''
            compute observed feature counts
            val = alpha_s * arc_weight * beta_t
            Where the arc goes from state s to state t
            '''
            val_obs = alpha_s.__mul__(a.weight).__mul__(beta_t)
            sparse_observed_counts[f_id] = sparse_observed_counts.get(f_id, fst.LogWeight.ZERO) + val_obs

    return sparse_expected_counts, sparse_observed_counts


def accumilate_counts(sparse_counts, global_counts):
    for k in sparse_counts:
        if k is not None:
            global_counts[k] += sparse_counts[k]
    return global_counts


def apply_weights(trellis, theta):
    #machine.write('before.fst', machine.isyms, machine.osyms)
    for s in trellis.states:
        for a in s.arcs:
            f_id = get_feature_id(a.ilabel, a.olabel, trellis)
            #print 'f_id', f_id, a.ilabel, a.olabel
            if f_id is not None:
                a.weight = fst.LogWeight(theta[f_id])
    #machine.write('after.fst', machine.isyms, machine.osyms)
    #exit()
    return trellis


def get_likelihood(obs_trellis, theta):
    obs_trellis.write('unnorm')
    obs_trellis = apply_weights(obs_trellis, theta)
    obs_trellis = renormalize(obs_trellis)
    obs_trellis.write('norm')
    try:
        ll = obs_trellis.shortest_distance(True)[0]
    except IndexError:
        print 'index error'
        return 0.0

    return float(ll)


def value(theta):
    likelihood = 0.0
    print 'likelihoods'
    for idx, obs_trellis_file in enumerate(obs_machines):
        sys.stdout.write('%d \r' % idx)
        sys.stdout.flush()
        obs_trellis = fst.read(path + obs_trellis_file)
        likelihood += get_likelihood(obs_trellis, theta)
    #reg = np.linalg.norm(theta, ord=1)
    print 'll', likelihood  #, 'reg', reg
    return likelihood


if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-l", "--fst-location", dest="fstLocation", default="fsts/", help="location of created fsts")
    optparser.add_option("-o", "--weights-location", dest="weightsLocation", default="fsts/", help="location of trained feature weights")
    (opts, _) = optparser.parse_args()
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
    sys.stdin = codecs.getwriter('utf-8')(sys.stdin)
    path = opts.fstLocation
    out_path = opts.weightsLocation + 'learned.features.'
    print 'reading data...'
    f_names = dict((tuple(n.split()[1].split('|||')), int(n.split()[0]) ) for n in codecs.open(path + 'E.names', 'r', 'utf-8').readlines())
    f_ids = dict((int(n.split()[0]), tuple(n.split()[1].split('|||'))) for n in codecs.open(path + 'E.names', 'r', 'utf-8').readlines())
    f_init_weights = [-float(n.split()[1]) for n in codecs.open(path + 'E.weights', 'r', 'utf-8').readlines()]
    inp_machines, obs_chain, exp_machines, obs_machines = zip(
        *[tuple(l.split()) for l in codecs.open(path + 'filenames', 'r').readlines()[1:]])

    inp_machines = ns(inp_machines)
    exp_machines = ns(exp_machines)
    obs_machines = ns(obs_machines)
    obs_chain = ns(obs_chain)

    #F = DifferentiableFunction(value, gradient)
    #(fopt, theta, return_status) = F.minimize(f_init_weights)
    #write_learned_features(theta)

    initial_theta = np.random.uniform(-1, 1, len(f_init_weights))
    #initial_theta = np.array(f_init_weights)
    eps = 0.01  # np.finfo(float).eps
    fprime = approx_fprime(initial_theta, value, [eps] * len(initial_theta))
    init_grad = gradient(initial_theta)
    cg = np.linalg.norm(fprime - init_grad, ord=2)
    print 'fprime, init_grad'
    pprint(zip(fprime, init_grad))
    print 'cg', cg
    #(xopt, fopt, return_status) = fmin_l_bfgs_b(value, initial_theta, gradient, pgtol=0.001)
    #write_learned_features(xopt)

    #wfst = fst.read('fsts/0.obs.fst')
    #wfst.write('unnorm', wfst.isyms, wfst.osyms)
    #re_wfst = renormalize(wfst)
    #re_wfst.write('norm', wfst.isyms, wfst.osyms)
