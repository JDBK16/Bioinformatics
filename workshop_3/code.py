import math
import random

# Question 1:
## Write log_odds_score(seq, M_plus, M_minus) matching the lecture's RATIO(x) formula, and a sliding-window scan(seq, M_plus, M_minus, window) that reports it across a long test sequence. Test on the supplied data sets to find CpG islands.

def log_odds_score(seq, M_plus, M_minus):
    seq = seq.upper()
    base = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3
    }
    
    ratio = 0
    for i in range(1, len(seq)):
        x_0 = base[seq[i-1]]
        x_1 = base[seq[i]]
        ratio += math.log(M_plus[x_0][x_1]/M_minus[x_0][x_1])
        
    return ratio

def scan(seq, M_plus, M_minus, window):
    window -= 1
    start = 0
    windows = []
    while window < len(seq):
        current_seq = seq[start:window]
        windows.append(log_odds_score(current_seq, M_plus, M_minus))
        start += 1
        window += 1
    return windows


# Copilot FASTA extractor
def read_fasta(filename):
    sequences = {}

    with open(filename) as f:
        current_id = None
        current_seq = []
        keys = []

        for line in f:
            line = line.strip()

            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)

                current_id = line[1:].split()[0]
                keys.append(current_id)
                current_seq = []
            else:
                current_seq.append(line)

        if current_id is not None:
            sequences[current_id] = "".join(current_seq)

    return keys, sequences

test_short_keys, test_short_sequences = read_fasta("../workshop_3/test_short.fasta")

M_plus = [[0.180, 0.274, 0.426, 0.120],
          [0.171, 0.368, 0.274, 0.188],
          [0.161, 0.339, 0.375, 0.125],
          [0.079, 0.355, 0.384, 0.182]]

M_minus = [[0.300, 0.205, 0.285, 0.210],
           [0.322, 0.298, 0.078, 0.302],
           [0.248, 0.246, 0.298, 0.208],
           [0.177, 0.239, 0.292, 0.292]]

test_short_scan = scan(test_short_sequences[test_short_keys[0]], M_plus, M_minus, 200)
    
class HMM:
    def __init__(self, states, init, trans, emit, symbols):
        self.states = states
        self.init = init
        self.trans = trans
        self.emit = emit
        self.symbols = symbols
        
    def simulate(self, length):
        start = random.random()
        initial = 0
        count = -1
        while initial < start:
            count += 1
            initial += self.init[count]
        
    def joint_prob(self):
        
        

stymbols = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3
}
states = {
    "M+": 0,
    "M-": 1
}
init = [0.25, 0.25, 0.25, 0.25]
emit = {
    "0": [[0.180, 0.274, 0.426, 0.120],
          [0.171, 0.368, 0.274, 0.188],
          [0.161, 0.339, 0.375, 0.125],
          [0.079, 0.355, 0.384, 0.182]],
    "1": [[0.300, 0.205, 0.285, 0.210],
          [0.322, 0.298, 0.078, 0.302],
          [0.248, 0.246, 0.298, 0.208],
          [0.177, 0.239, 0.292, 0.292]]
}
trans = [[1-0.01, 0.01],
         [0.001, 1-0.001]]

HMM = HMM(states, init, trans, emit, symbols)
HMM.simulate(2)