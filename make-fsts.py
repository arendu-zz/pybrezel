__author__ = 'arenduchintala'
import fst
import itertools as it
from multiprocessing import Pool
import multiprocessing
from math import log
import os, optparse

global global_features
global_features = set([])
global V
V = set([])
global filenames
filenames = []


def get_weight(feat, emit, co):
    # done by taking the first token after comma separating feat
    f = feat.split(',')[0]
    e = co[f]
    if emit in e:
        return log(1.0 / len(e))
    else:
        return float('-inf')


def limited_distortion(feature_labels, len_tar, sym_features, sym_target, co_oc):
    global V
    d_lim = fst.LogTransducer(sym_features, sym_features)
    for i in range(len_tar):
        for feature in feature_labels:
            # for countv, v in co_oc[feature]:
            d_lim.add_arc(i, i + 1, feature, feature, 0.0)
    d_lim[i + 1].final = True
    return d_lim


def get_feature(feature_type, from_id, to_id, from_token, to_token, emit_token=None):
    if feature_type == 'model1':
        return from_token
    elif feature_type == 'hmm':  # HMM feature with relative jump distance
        return from_token + '|||' + str(abs(from_id - to_id))
    elif feature_type == 'allfeatures':
        return from_token + '|||' + str(from_id) + '|||' + to_token + '|||' + str(to_id)
    else:
        return from_token + '|||' + to_token


def accumilate_features(results):
    global global_features
    global filenames
    print 'accumulating results'
    features_seen = results[0]
    names = results[1]
    filenames.append(names)
    global_features.update(features_seen)


def save_fst(idx, src, tar, save_path, feature_type, co_oc):
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
    trans_fst.write(save_path + str(idx) + '.trans.fst', sym_features, sym_features)
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

    # print 'saved files.. starting composition...'
    write_E(save_path, str(idx), feature_labels_seen, co_oc)

    cmd = 'fstcompose ' + save_path + str(idx) + '.inp.fst ' + save_path + str(idx) + '.E.fst > ' + save_path + str(idx) + '.exp.fst'
    os.system(cmd)
    cmd1 = 'fstcompose ' + save_path + str(idx) + '.exp.fst ' + save_path + str(idx) + '.y.fst > ' + save_path + str(idx) + '.obs.fst'
    os.system(cmd1)
    names = str(idx) + '.inp.fst' + '\t' + str(idx) + '.y.fst' + '\t' + str(idx) + '.exp.fst' + '\t' + str(idx) + '.obs.fst'
    return feature_labels_seen, names


def write_E(save_path, ename, feature_labels_seen, co_oc):
    writer_features = open(save_path + ename + '.E.names', 'w')
    writer_weights = open(save_path + ename + '.E.weights', 'w')
    writer_ids = open(save_path + ename + '.E.ids', 'w')
    E_fst = fst.LogTransducer(sym_features, sym_targets)
    idx = 0
    for feat in feature_labels_seen:
        for countv, v in co_oc[feat]:
            idx += 1
            E_fst.add_arc(0, 0, feat, v, 0.0)
            writer_features.write(str(idx) + '\t' + feat + ':' + v + '\n')
            writer_weights.write(str(idx) + '\t' + str(0.0) + '\n')
            writer_ids.write(str(idx) + '\t' + str(idx) + '\n')
    E_fst[0].final = True
    E_fst.write(save_path + ename + '.E.fst', sym_features, sym_targets)
    writer_weights.flush()
    writer_features.flush()
    writer_ids.flush()
    writer_weights.close()
    writer_features.close()
    writer_ids.close()


if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-s", "--source", dest="source", default="data/toy1/en", help="source file")
    optparser.add_option("-t", "--target", dest="target", default="data/toy1/fr", help="target file")
    optparser.add_option("-f", "--feature-type", dest="featureType", default="allfeatures", help="encode model1 or hmm features")
    optparser.add_option("-l", "--fst-location", dest="fstLocation", default="fsts/", help="Where to save the created fsts")
    optparser.add_option("-j", "--jump-width", dest="jumpWidth", default=100, type="int", help="span width to count co-occurrence")
    (opts, _) = optparser.parse_args()
    sym_features = fst.SymbolTable()
    sym_features[fst.EPSILON] = 0
    sym_targets = fst.SymbolTable()
    sym_targets[fst.EPSILON] = 0
    jump_limit = opts.jumpWidth  # float('inf')
    save_path = opts.fstLocation
    feature_type = opts.featureType
    source_sentences = open(opts.source).readlines()
    target_sentences = open(opts.target).readlines()

    global_features = set([])

    co_occurrence = {}
    src_list = []
    tar_list = []
    print 'collecting co-occurrence...'
    for idx, (src_sent, tar_sent) in enumerate(zip(source_sentences, target_sentences)):
        tl = tar_sent.split()
        sl = src_sent.split()
        sl.insert(0, 'NULL')
        V.update(tl)
        for (from_id, to_id) in it.product(range(len(sl)), range(len(sl))):
            if abs(from_id - to_id) > jump_limit:
                continue
            from_token = sl[from_id]
            to_token = sl[to_id]
            from_id += 1
            to_id += 1  # zero is saved for the start state
            feature_label = get_feature(feature_type, from_id, to_id, from_token, to_token)
            sym_features[feature_label]
            co = co_occurrence.get(feature_label, {})
            # co += list(set(tl))  # count how many times it was seen together per sentence but not within a sentence
            for t in set(tl):  # count how many times it was seen together per sentence but not within a sentence
                co[t] = co.get(t, 0) + 1
                sym_targets[t]
            co_occurrence[feature_label] = co
        src_list.append(sl)
        tar_list.append(tl)

    print 'sorting co-occurrence...'
    for idx, k in enumerate(co_occurrence):
        dict_kc = co_occurrence[k]
        count_kc = [(countv, v) for v, countv in dict_kc.items()]
        if len(count_kc) > 0:
            max_co = max(count_kc)[0]
            min_co = min(count_kc)[0]
            # major hack
            # if max_co - min_co > 300 and k != 'NULL':
            # bottom_percent = min_co + (0.2 * (max_co - min_co))
            # count_kc = [(countv, v) for (countv, v) in count_kc if countv >= bottom_percent]
            co_occurrence[k] = count_kc

    pool = Pool(processes=multiprocessing.cpu_count())
    for idx, (src, tar) in enumerate(zip(src_list, tar_list)):
        # print src, '\n', tar
        pool.apply_async(save_fst, args=(idx, src, tar, save_path, feature_type, co_occurrence), callback=accumilate_features)
    pool.close()
    pool.join()
    # write_E(save_path, 'FullE', global_features, co_occurrence)
    writer_filenames = open(save_path + 'filenames', 'w')
    writer_filenames.write(str(len(filenames)) + '\n')
    writer_filenames.write('\n'.join(sorted(filenames)))
    writer_filenames.flush()
    writer_filenames.close()
    writer_features = open(save_path + 'E.names', 'w')

    writer_int_names = open(save_path + 'E.intnames', 'w')
    writer_weights = open(save_path + 'E.weights', 'w')
    writer_ids = open(save_path + 'E.ids', 'w')
    E_fst = fst.LogTransducer(sym_features, sym_targets)
    I_fst = fst.LogTransducer(sym_features, sym_features)
    I_fst[0].final = True
    I_fst.add_arc(0, 0, fst.EPSILON, fst.EPSILON, 0.0)
    I_fst.write(save_path + "I.fst", sym_features, sym_features)
    idx = 0
    for feat in global_features:
        for countv, v in co_occurrence[feat]:
            E_fst.add_arc(0, 0, feat, v, 0.0)
            writer_features.write(str(idx) + '\t' + feat + '|||' + v + '\n')
            writer_int_names.write(str(idx) + '\t' + str(sym_features[feat]) + '\t' + str(sym_targets[v]) + '\n')
            writer_weights.write(str(idx) + '\t' + str(0.0) + '\n')
            writer_ids.write(str(idx) + '\t' + str(idx) + '\n')
            idx += 1
    E_fst[0].final = True
    E_fst.write(save_path + 'E.fst', sym_features, sym_targets)
    writer_weights.flush()
    writer_features.flush()
    writer_int_names.flush()
    writer_ids.flush()
    writer_weights.close()
    writer_features.close()
    writer_int_names.close()
    writer_ids.close()


