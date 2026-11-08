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
    def __init__(self, states, init, trans, emit):
        self.states = states
        self.init = init
        self.trans = trans
        self.emit = emit
        
    def sample(self, dist):
        rand = random.random()
        cum = 0
        last = None
        for letter, p in dist.items():
            last = letter
            if cum > rand:
                return last
            cum += p
        return last
            
    def simulate(self, length):
        state = self.sample(self.init)
        path, symbols = [], []
        for i in range(length):
            if i > 0:
                state = self.sample(self.trans[state])
            path.append(state)
            symbols.append(self.sample(self.emit[state]))
        return path, symbols
    
    def joint_prob(self, states, symbols):
        p = self.init[states[0]] * self.emit[states[0]][symbols[0]]
        for i in range(1, len(states)):
            p *= self.trans[states[i-1]][states[i]]*self.emit[states[i]][symbols[i]]
        return p

        

letters = ["A", "C", "G", "T"]

M_plus = {
    "A": {"A": 0.180, "C": 0.274, "G": 0.426, "T": 0.120},
    "C": {"A": 0.171, "C": 0.368, "G": 0.274, "T": 0.188},
    "G": {"A": 0.161, "C": 0.339, "G": 0.375, "T": 0.125},
    "T": {"A": 0.079, "C": 0.355, "G": 0.384, "T": 0.182},
}
M_minus = {
    "A": {"A": 0.300, "C": 0.205, "G": 0.285, "T": 0.210},
    "C": {"A": 0.322, "C": 0.298, "G": 0.078, "T": 0.302},
    "G": {"A": 0.248, "C": 0.246, "G": 0.298, "T": 0.208},
    "T": {"A": 0.177, "C": 0.239, "G": 0.292, "T": 0.292},
}

p = 0.01    # leave CPG
q = 0.001   # enter CPG

states = [b + "+" for b in letters] + [b + "-" for b in letters]

trans = {}
for x in letters:
    trans[x + "+"] = {}
    trans[x + "-"] = {}
    for y in letters:
        trans[x + "+"][y + "+"] = (1 - p) * M_plus[x][y]
        trans[x + "+"][y + "-"] = p * M_minus[x][y]
        trans[x + "-"][y + "-"] = (1 - q) * M_minus[x][y]
        trans[x + "-"][y + "+"] = q * M_plus[x][y]

emit = {}
for s in states:
    emit[s] = {b: (1.0 if b == s[0] else 0.0) for b in letters}

init = {s: 1 / 8 for s in states}

cpg_hmm = HMM(states, init, trans, emit)

path, symbols = cpg_hmm.simulate(10)
print(path)
print(symbols)