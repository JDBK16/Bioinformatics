def sigma(a, b):
    """Scoring function: match +1, mismatch -1, indel -2."""
    if a == "-" or b == "-":
        return -2
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
    return V[-1][-1][0], top + "\n" + mid + "\n" + bot
 
 
if __name__ == "__main__":
    s1, s2 = "GCGCATGGATTGAGCGA", "TGCGCCATTGATGACCA"
    score, alignment = align(s1, s2)
    print("Optimal score:", score)
    print(alignment)