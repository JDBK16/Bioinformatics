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
        
    print("s backwards:", s[d_s:i])
    print("s forwards:", s[i:x_s+1])
    print("t backwards:", t[d_t:j])
    print("t forwards:", t[j:x_t+1])
            

s = "AGTCGATCGAGTCGA"
t = "AGTCGATCGAGTCGA"
map = cmer_index(s, 4)
seed_pairs = find_seeds(map, t, 4)
histogram, max_diag = diagonal_histogram(seed_pairs)

print(map)
print(seed_pairs)
print(histogram)
print(max_diag)

ungapped_extend(s, t, 6, 6, 3)