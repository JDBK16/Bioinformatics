import pandas as pd

def sigma(a, b):
    """Scoring function: match +1, mismatch -1, indel -2."""
    if a == "-" or b == "-":
        return -1
    if a == b:
        return 1
    return -1
 
 
def align(s, t):
    """Global alignment of s and t by the Needleman-Wunsch algorithm.
 
    Returns (optimal_score, formatted_alignment_string).
    """
    V = [[None for _ in "-" + t] for _ in "-" + s]
    V[0][0] = (0, "fin")
    for i, a in enumerate(s, start=1):
        V[i][0] = (V[i - 1][0][0] + sigma(a, "-"), "up")
    for j, b in enumerate(t, start=1):
        V[0][j] = (V[0][j - 1][0] + sigma("-", b), "left")
 
    for i, a in enumerate(s, start=1):
        for j, b in enumerate(t, start=1):
            opts = {
                "up":   V[i - 1][j][0]   + sigma(a, "-"),
                "diag": V[i - 1][j - 1][0] + sigma(a, b),
                "left": V[i][j - 1][0]   + sigma("-", b),
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

# Copilot improved
def print_matrix(s, t):
    V, _, _ = align(s, t)

    rows = [""] + list(s)
    cols = [""] + list(t)
    
    print("", end="\t")
    for col in cols:
        print(col, end="\t")
    print()
    
    for i in range(len(V)):
        print(rows[i], end="\t")

        for j in range(len(V[i])):
            score, direction = V[i][j]
            if direction == "left":
                arrow = "←"
            elif direction == "up":
                arrow = "↑"
            elif direction == "diag":
                arrow = "↖"
            else:
                arrow = ""
            print(f"{score}{arrow}", end="\t")
        print()
        
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

def pairwise_align(keys, fasta):
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            print(keys[i], "VS", keys[j] + ":")
            _, score, alignment = align(fasta[keys[i]], fasta[keys[j]])
            print("Optimal score:", score)
            # print(alignment)



def local_align(s, t):
    V = [[None for _ in "-" + t] for _ in "-" + s]
    V[0][0] = (0, "fin")
    for i, a in enumerate(s, start=1):
        V[i][0] = (0, "up")
    for j, b in enumerate(t, start=1):
        V[0][j] = (0, "left")

    for i, a in enumerate(s, start=1):
        for j, b in enumerate(t, start=1):
            opts = {
                "zero": 0,
                "up":   V[i - 1][j][0] + sigma(a, "-"),
                "diag": V[i - 1][j - 1][0] + sigma(a, b),
                "left": V[i][j - 1][0] + sigma("-", b),
            }
            choice = max(opts, key=lambda k: opts[k])
            V[i][j] = (opts[choice], choice)

    max_coords = (0, 0)
    max_score = 0
    for i in range(len(s) + 1):
        for j in range(len(t) + 1):
            if V[i][j][0] > max_score:
                max_coords = (i, j)
                max_score = V[i][j][0]

    # Traceback from maximum
    i, j = max_coords
    pairs = []
    while V[i][j][0] != 0:
        direction = V[i][j][1]
        if direction == "up":
            pairs.append((s[i - 1], "-"))
            i -= 1
        elif direction == "diag":
            pairs.append((s[i - 1], t[j - 1]))
            i -= 1
            j -= 1
        elif direction == "left":
            pairs.append(("-", t[j - 1]))
            j -= 1
    pairs.reverse()

    top = " ".join(p[0] for p in pairs)
    mid = " ".join("|" if p[0] == p[1] else " " for p in pairs)
    bot = " ".join(p[1] for p in pairs)

    return V, max_score, top + "\n" + mid + "\n" + bot
 
 
if __name__ == "__main__":
    #s1, s2 = "GCGCATGGATTGAGCGA", "TGCGCCATTGATGACCA"
    s1 ='TTCCAATCTCGAGATCGGATC'
    s2 ='GATCGGATCTTGGCCTAGCTA'
    V, score, alignment = align(s1, s2)
    print("Optimal global score:", score)
    print(alignment)
    # print_matrix(s1, s2)
    # keys, fasta = read_fasta("02_globin_cds.fasta")
    # pairwise_align(keys, fasta)
    V, score, alignment = local_align(s1, s2)
    print("Optimal local score:", score)
    print(alignment)
    