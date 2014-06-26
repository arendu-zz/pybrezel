__author__ = 'arenduchintala'
global pairs, expectedCounts, decisionToContext, conditionals
pairs = []
expectedCounts = {}  # expected counts
conditionals = {}  # conditional probabilities p(d | c) at current feature weights
decisionToContext = {}  # all contexts for a particular decision


def main():
    for c, d in expectedCounts:
        # c,d = context,decision
        print c, d


if __name__ == '__main__':
    main()