import random
import math

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
    
    def viterbi(self, symbols):
        L = len(symbols)
        v = [{} for _ in range(L)]
        prev = [{} for _ in range(L)]
        
        for l in self.states:
            v[0][l] = math.log(self.init[l]) + math.log(self.emit[l][symbols[0]])
        
        for i in range(1, L):
            for l in self.states:
                best_k, best_val = None, -math.inf
                for k in self.states:
                    val = v[i-1][k] + math.log(self.trans[k][l])
                    if best_k is None or val > best_val:
                        best_k, best_val = k, val
                v[i][l] = math.log(self.emit[l][symbols[i]]) + best_val
                prev[i][l] = best_k
        
        last = max(self.states, key=lambda l: v[L-1][l])
        best_log = v[L-1][last]
        path = [last]
        for i in range(L-1, 0, -1):
            path.append(prev[i][path[-1]])
        path.reverse()
        return path, math.exp(best_log) 
    
    def forward(self, symbols):
        L = len(symbols)
        F = [{} for _ in range(L)]
        for l in self.states:
            F[0][l] = math.log(self.init[l]) + math.log(self.emit[l][symbols[0]])
        for i in range(1, L):
            for l in self.states:
                s = math.log(sum(math.exp([F[i-1][k] + math.log(self.trans[k][l]) for k in self.states])))
                F[i][l] = math.log(self.emit[l][symbols[i]]) + s
        
        logP = math.log(sum(math.exp([F[L-1][k] for k in self.states])))
        return F, logP
    
    def backward(self, symbols):
        L = len(symbols)
        B = [{} for _ in range(L)]
        for l in self.states:
            B[L-1][1] = 0.0
        for i in range(L-2, -1, -1):
            for l in self.states:
                B[i][l] = math.log(sum(math.exp([
                    math.log(self.trans[l][k]) + math.log(self.emit[k][symbols[i+1]] + B[i+1][k]
                                                          for k in self.states)
                ])))
        return B
    
    
coin = HMM(
    states=["F", "B"],
    init={"F": 0.5, "B": 0.5},
    trans={"F": {"F": 0.75, "B": 0.25}, "B": {"F": 0.25, "B": 0.75}},
    emit={"F": {"H": 0.5, "T": 0.5}, "B": {"H": 0.75, "T": 0.25}},
)

path, best = coin.viterbi(list("TTHHT"))
print(path, best)
print(coin.forward("TTHHT"))