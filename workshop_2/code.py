import math

#Write cmer_index(seq, c) returning a dictionary mapping each length-c word to the (1-indexed) list of positions where it occurs in seq.
def cmer_index(seq, c):
    map = {}
    if c > len(seq):
        return map
    start = 0
    while start + c < len(seq):
        word = seq[start:start+c]
        if word in map:
            map[word].append(start)
        else:
            map[word] = ["-", start]
        start += 1
    return map

#Write find_seeds(index_table, query, c) that slides a c-window along query, looks each word up, and returns a list of (i, j) seed pairs. Then write diagonal_histogram(seeds) that groups seeds by diagonal (i – j) and reports the diagonal(s) with the most support.

def find_seeds(index_table, query, c):
    seed_pairs = []
    query_map = cmer_index(query, c)
    for q_key, q_items in query_map.items():
        if q_key in index_table:
            for i in range(1, len(q_items)):
                for j in range(1, len(index_table[q_key])):
                    seed_pairs.append((q_items[i], index_table[q_key][j]))
    
    return seed_pairs

def diagonal_histogram(seeds):
    histogram = {}
    for (i, j) in seeds:
        if i - j in histogram:
            histogram[i-j].append((i, j))
        else:
            histogram[i-j] = [(i, j)]
    max_key = 0
    max_len = -1
    for key, value in histogram.items():
        if len(value) > max_len:
            max_len = len(value)
            max_key = key
    
    return histogram, max_key

# Write ungapped_extend(s, t, i, j, drop_threshold=3) that, given a seed match at (i, j), extends it in both directions along the diagonal, keeping a running score,stopping (in each direction) once the score falls more than some threshold below its best value so far (say 3 below best).

def ungapped_extend(s, t, i, j, drop_threshold=3):
    s_end = i
    t_end = j
    while True:
        if s_end < len(s) - 1 and t_end < len(t) - 1:
            if s[s_end + 1] == t[t_end + 1]:
                s_end += 1
                t_end += 1
            else:
                break
        else:
            break
    
    extend_score = 0
    x_s = s_end
    x_t = t_end
    while True:
        if x_s < len(s) - 1 and x_t < len(t) - 1:
            if s[x_s + 1] == t[x_t + 1]:
                extend_score += 1
            else:
                extend_score -= 1
                if extend_score <= -3:
                    break
            x_s += 1
            x_t += 1
        else:
            break
        
    extend_score = 0
    d_s = i
    d_t = j
    while True:
        if d_s > 0 and d_t > 0:
            if s[d_s - 1] == t[d_t - 1]:
                extend_score += 1
            else:
                extend_score -= 1
                if extend_score <= -3:
                    break
            d_s -= 1
            d_t -= 1
        else:
            break
    return (x_s + x_t - d_s - d_t)
    
def sigma(a, b):
    """Scoring function: match +1, mismatch -1, indel -2."""
    if a == "-" or b == "-":
        return -1
    if a == b:
        return 1
    return -1

def banded_aligns(s, t, k, offset=0):
    V = [[None for _ in "-" + t] for _ in "-" + s]
    V[0][0] = (0, "fin")
    for i, a in enumerate(s, start=1):
        if abs(i - 0 - offset) <= k//2:
            V[i][0] = (V[i - 1][0][0] + sigma(a, "-"), "up")
        else:
            V[i][0] = (-math.inf, "up")
    for j, b in enumerate(t, start=1):
        if abs(0 - j - offset) <= k//2:
            V[0][j] = (V[0][j - 1][0] + sigma("-", b), "left")
        else:
            V[0][j] = (-math.inf, "left")
    for i, a in enumerate(s, start=1):
        for j, b in enumerate(t, start=1):
            if abs(i - j - offset) <= k//2:
                up = V[i - 1][j][0]   + sigma(a, "-")
                left = V[i][j - 1][0]   + sigma("-", b)
                diag = V[i - 1][j - 1][0] + sigma(a, b)
            else:
                up = left = diag = -math.inf
                 
            opts = {
                "up":   up,
                "diag": diag,
                "left": left,
            }
            choice = max(opts, key=lambda k: opts[k])
            V[i][j] = (opts[choice], choice)
    
    # Traceback
    s_iter, t_iter = iter(s[::-1] + "-"), iter(t[::-1] + "-")
    i, j = len(s), len(t)
    pairs = []
    while i > 0 or j > 0:
        direction = V[i][j][1]
        if direction == "up":
            pairs.append((next(s_iter), "-")); i -= 1
        elif direction == "diag":
            pairs.append((next(s_iter), next(t_iter))); i -= 1; j -= 1
        elif direction == "left":
            pairs.append(("-", next(t_iter))); j -= 1
    pairs.reverse()
    top = " ".join(p[0] for p in pairs)
    mid = " ".join("|" if p[0] == p[1] else " " for p in pairs)
    bot = " ".join(p[1] for p in pairs)
    return V, V[-1][-1][0], top + "\n" + mid + "\n" + bot


def entire(s, t, len):
    map = cmer_index(s, len)
    seed_pairs = find_seeds(map, t, len)
    hist, max_diag = diagonal_histogram(seed_pairs)
    best = 0
    best_ij = hist[max_diag][0]
    for i, j in hist[max_diag]:
        ext = ungapped_extend(s, t, i, j, 3)
        if ext >= best:
            best = ext
            best_ij = (i, j)
    return banded_aligns(s, t, 6, (i-j))

        
    
s = "GGGAGTCGATCGAGTCGA"
t = "AGTCGATCGAGTCGA"
V, score, alignment = entire(s, t, 4)
print(score, "\n", alignment)
"""
map = cmer_index(s, 4)
seed_pairs = find_seeds(map, t, 4)
histogram, max_diag = diagonal_histogram(seed_pairs)

print(map)
print(seed_pairs)
print(histogram)
print(max_diag)

ungapped_extend(s, t, 6, 6, 3)

V, score, alignment = banded_aligns("ATATATAGTTATATA", "ATATATATTTATATA", 10, 0)
print(V)
print(score),
print(alignment)"""