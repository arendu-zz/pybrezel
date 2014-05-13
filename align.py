# -*- coding: utf-8 -*-
__author__ = 'arenduchintala'
import fst, sys, codecs, os, pdb
import re

global sym_features, sym_targets


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def make_fst(src, tar):
    print idx, 'sent pair'
    trans_fst = fst.LogTransducer(sym_features, sym_features)
    feature_labels_seen = []
    for (from_id, to_id) in it.product(range(len(src)), range(len(src))):
        if abs(from_id - to_id) > jump_limit:
            continue
        from_token = src[from_id]
        to_token = src[to_id]
        from_id += 1
        to_id += 1  # zero is saved for the start state
        feature_label = get_feature(feature_type, from_id, to_id, from_token, to_token)
        feature_labels_seen.append(feature_label)
        trans_fst.add_arc(from_id, to_id, feature_label, feature_label, 0.0)

    for f in range(1, len(src) + 1):
        trans_fst.add_arc(0, f, fst.EPSILON, fst.EPSILON, 0.0)
        trans_fst[f].final = True
    trans_fst.arc_sort_input()
    feature_labels_seen = set(feature_labels_seen)
    d = limited_distortion(feature_labels_seen, len(tar), sym_features, sym_features, co_oc)
    d.arc_sort_input()
    d.write(save_path + str(idx) + '.d.fst', sym_features, sym_features)
    inp = trans_fst.compose(d)
    inp.write(save_path + str(idx) + '.inp.fst', sym_features, sym_features)
    y = fst.LogTransducer(sym_targets, sym_targets)
    for y_idx, t in enumerate(tar):
        y.add_arc(y_idx, y_idx + 1, t, t, 0.0)
    y[y_idx + 1].final = True
    y.write(save_path + str(idx) + '.y.fst', sym_targets, sym_targets)

    #print 'saved files.. starting composition...'
    write_E(save_path, str(idx), feature_labels_seen, co_oc)

    cmd = 'fstcompose ' + save_path + str(idx) + '.inp.fst ' + save_path + str(idx) + '.E.fst > ' + save_path + str(idx) + '.exp.fst'
    os.system(cmd)
    cmd1 = 'fstcompose ' + save_path + str(idx) + '.exp.fst ' + save_path + str(idx) + '.y.fst > ' + save_path + str(idx) + '.obs.fst'
    os.system(cmd1)
    names = str(idx) + '.inp.fst' + '\t' + str(idx) + '.y.fst' + '\t' + str(idx) + '.exp.fst'
    return feature_labels_seen, names


def get_feature_id(isym_id, osym_id, f):
    r = f_names.get((f.isyms.find(isym_id), f.osyms.find(osym_id)), None)
    return r


def apply_weights(machine, weights):
    #machine.write('before.fst', machine.isyms, machine.osyms)
    for s in machine.states:
        for a in s.arcs:
            f_id = get_feature_id(a.ilabel, a.olabel, machine)
            if f_id is not None:
                a.weight = fst.LogWeight(weights[f_id])
            elif a.ilabel == 0 and a.olabel == 0:
                pass
            else:
                print 'no f id', a.ilabel, a.olabel, machine.isyms.find(a.ilabel), machine.osyms.find(a.olabel)
    #machine.write('after.fst', machine.isyms, machine.osyms)
    #exit()
    return machine


def do_align(idx, best, s, t):
    alignments = []
    for path in best.paths():  # refactor out the loop
        path_istring = [best.isyms.find(arc.ilabel) for arc in path if best.isyms.find(arc.ilabel) != fst.EPSILON]
        path_ostring = [best.osyms.find(arc.olabel) for arc in path if best.osyms.find(arc.olabel) != fst.EPSILON]
        for i, (s_token, t_token) in enumerate(zip(path_istring, path_ostring)):
            s_token = s_token.split(',', 1)[-1]
            if s_token != 'NULL':
                matches = sorted([(abs(i_st - i), i_st) for i_st, st in enumerate(s) if st == s_token])
                b_match = matches[0][1]
                alignments.append(' '.join([str(idx), str(b_match + 1), str(i + 1)]))
            else:
                pass
                #alignments.append(' '.join([str(idx), str(0), str(i + 1)]))
    return alignments


if __name__ == '__main__':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
    sys.stdin = codecs.getwriter('utf-8')(sys.stdin)
    path = '/Users/arenduchintala/PycharmProjects/alignment-FSTs/coursera-20/train-20-hmm/'
    #path = 'data/toy2/'
    learned_weight_file = 'E.from.m1.init.weights' #path + 'learned.features.4'
    f_names = dict((tuple(n.split()[1].split('|||')), int(n.split()[0])) for n in codecs.open(path + 'E.names', 'r', 'utf-8').readlines())
    f_ids = dict((int(n.split()[0]), tuple(n.split()[1].split('|||'))) for n in codecs.open(path + 'E.names', 'r', 'utf-8').readlines())
    f_init_weights = dict((int(n.split()[0]), -float(n.split()[1])) for n in codecs.open(path + 'E.weights', 'r', 'utf-8').readlines())

    learned_weights = dict(
        (int(l.split('\t')[0]), float(l.split('\t')[-1])) for l in codecs.open(learned_weight_file, 'r', 'utf-8').readlines())

    filenames = codecs.open(path + 'filenames', 'r', 'utf-8').readlines()[1:]
    nat_sort_filenames = natural_sort(filenames)
    inp_machines, obs_chain, exp_machines = zip(*[tuple(l.split()) for l in nat_sort_filenames])

    obs_trelis = [o.replace('y', 'obs') for o in obs_chain]
    source = [l.split() for l in codecs.open(path + 'en', 'r', 'utf-8').readlines()]
    target = [l.split() for l in codecs.open(path + 'fr', 'r', 'utf-8').readlines()]

    all_alignments = []
    for idx, (ot, s, t) in enumerate(zip(obs_trelis, source, target)[:53]):
        print idx, s, t
        obs_t = fst.read(path + ot)
        sym_features = obs_t.isyms
        sym_targets = obs_t.osyms
        print path + ot
        obs_t.write('obs_t.fst')
        obs_wt = apply_weights(obs_t, learned_weights)
        obs_wt.write('obs_wt.fst', obs_t.isyms, obs_t.osyms)
        os.system('fstmap --map_type="to_standard" obs_wt.fst > obs_wt.std.fst')
        obs_wt_std = fst.read('obs_wt.std.fst')
        best_path = obs_wt_std.shortest_path()
        best_path.write('best_path.fst', obs_t.isyms, obs_t.osyms)
        all_alignments += do_align(idx + 1, best_path, s, t)
    writer = codecs.open('never.gonna.work.20.alignments.out', 'w')
    writer.write('\n'.join(all_alignments))
    writer.flush()
    writer.close()
