__author__ = 'arenduchintala'
import fst


class SentencePair:
    def __init__(self, M, Y, E, x):
        self.M = M  # FST representing trellis
        self.Y = Y  # FSA representing source word transitions (for this source sentence)
        self.E = E  # FST of word translations
        self.x = x  # linear chain FSA of target sentence
        self.e = {}  # dictionary holding expectation counts of features fired in the sent pair
        self.dc = {}  # dictionary holding the tuple (decision,type) (as key) and a list of contexts that it appears with

    def getFeatureDecomposition(self, arc):
        '''
        Takes in a tuple representing the arc in M and returns the
        origin of this arc in the Y machine and E machine.
        Y machine is the transition machine
        E machine is the emission machine
        :param arcLabel:
        :return: arc_transtion, arc_emission
        '''
        a_t = (arc.ilabel, arc.nextstate)
        a_e = (arc.ilabel, arc.olabel)
        return a_t, a_e

    def computeExpectation(self):
        self.M = fst.LogTransducer()
        alpha = self.M.shortest_distance()
        beta = self.M.shortest_distance(True)
        isyms = self.M.isyms
        osyms = self.M.osyms
        for s in self.M.states:
            for a in s.arcs:
                alpha_s = alpha[s.stateid]
                beta_s = beta[s.stateid]
                beta_t = beta[a.nextstate]
                a_weight = a.weight
                from_state = isyms.find(a.ilabel).split('|||')[0]
                to_state = isyms.find(a.ilabel).split('|||')[1]
                arc_exp = alpha_s.__mul__(a_weight).__mul__(beta_t).__div__(beta[0])
                current_wt = self.e.get((from_state, osyms.find(a.olabel)), fst.LogWeight.ZERO)
                current_wt = current_wt.__add__(arc_exp)
                '''
                emission feature type
                '''
                self.e[isyms.find(a.ilabel), osyms.find(a.olabel)] = current_wt
                '''
                adding context of the emission decision into dc dictionary
                '''
                contexts = self.dc.get((osyms.find(a.olabel), 'emission'), set([]))
                contexts.add((isyms.find(a.ilabel), osyms.find(a.olabel)))
                self.dc[(osyms.find(a.olabel), 'emission')] = contexts
                '''
                transition feature type
                '''
                current_wt = self.e.get((from_state, to_state), fst.LogWeight.ZERO)
                current_wt = current_wt.__add__(arc_exp)
                self.e[from_state, to_state] = current_wt
                '''
                adding context of transition decision into dc dict
                '''
                contexts = self.dc.get((from_state, 'transition'), set([]))
                contexts.add((from_state, to_state))
                self.dc[from_state, 'transition'] = contexts
