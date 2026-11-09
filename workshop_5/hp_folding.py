"""
Workshop 5 : HP-Lattice Folding

Fill in every function marked TODO. Run this file to see your progress
(sections whose functions aren't written yet are skipped)

Conventions
-----------
* A sequence is a string of 'H' and 'P' characters, e.g. "HPPH".
* Residue 0 is at (0, 0) and residue 1 at (1, 0). 
* A conformation of an n-residue chain is a list (or string) of n-2 turns,
  each 'F' (forward), 'L' (left, anticlockwise) or 'R' (right, clockwise),
  relative to the direction of the previous step.
* Lattice points / coordinates are tuples (x, y).
"""

import math
import random
import time
from itertools import product

TURNS = ("F", "L", "R")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def turn(direction, t):
    """If the last step was in direction direction, return the next step after applying turn t ('F', 'L' or 'R')    """
    # TODO
    raise NotImplementedError


def are_adjacent(p, q):
    """True if lattice points p and q are lattice neighbours"""
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Task 1
# ---------------------------------------------------------------------------
def generate_coords(turns):
    coords = [(0, 0), (1, 0)]
    direction = (1, 0)
    rotate = {
        (1, 0): {
            "L": (0, -1),
            "F": (1, 0),
            "R": (0, 1)
        },
        (0, 1): {
            "L": (1, 0),
            "F": (0, 1),
            "R": (-1, 0)
        },
        (-1, 0): {
            "L": (0, 1),
            "F": (-1, 0),
            "R": (0, -1)
        },
        (0, -1): {
            "L": (-1, 0),
            "F": (0, -1),
            "R": (1, 0)
        }
    }
    for turn in turns:
        direction = rotate[direction][turn]
        position = (coords[-1][0] + direction[0], coords[-1][1] + direction[1])
        coords.append(position)
    return coords


# ---------------------------------------------------------------------------
# Task 2
# ---------------------------------------------------------------------------
def is_valid_walk(coords):
    hash_table = {0: [(0, 0)]}
    prev = (0, 0)
    for row, col in coords:
        if row != 0 or col != 0:
            hash = row + col
            coord = (row, col)
            if hash in hash_table:
                if coord in hash_table[hash]:
                    return False
                hash_table[hash].append(coord)
            else:
                hash_table[hash] = [coord]
                
            if (abs(row-prev[0]) == 1 and col == prev[1]) or (abs(col-prev[1]) == 1 and row == prev[0]):
                prev = coord
            else:
                return False
        else:
            prev = (0, 0)
    return True


# ---------------------------------------------------------------------------
# Task 3
# ---------------------------------------------------------------------------
def energy(sequence, coords):
    """HP contact energy of the conformation given by coords. Energy: -1 for every pair i < j with s[i] == s[j] == 'H', |i - j| > 1,
    and coords[i], coords[j] lattice neighbours.

    Easy: direct O(n^2) double loop over pairs i < j.
    Challenge: an O(n) version using a dictionary {coordinate: index} each residue checks only its four lattice neighbours.
    """
    energy = 0
    dic = {}
    for index, coord in enumerate(coords):
        dic[coord] = index
    
    for index, coord in enumerate(coords):
        if sequence[index] != "H":
            dic.pop(coord, None)
        else:
            c1 = (coord[0] + 1, coord[1])
            c2 = (coord[0] - 1, coord[1])
            c3 = (coord[0], coord[1] + 1)
            c4 = (coord[0], coord[1] - 1)
            check = []
            if c1 in dic: check.append(dic[c1])
            if c2 in dic: check.append(dic[c2])
            if c3 in dic: check.append(dic[c3])
            if c4 in dic: check.append(dic[c4])
            for i in check:
                if sequence[i] == "H":
                    energy -= 1
            dic.pop(coord, None)
    return energy
                
                
def turns_energy(sequence, turns):
    """Energy of a turns list, or math.inf if the walk is not self-avoiding. (Handy for Task 7.)"""
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Task 4: brute-force enumeration
# ---------------------------------------------------------------------------
def enumerate_conformations(sequence):
    """Enumerate every self-avoiding walk of the sequence on the 2D square lattice (first step fixed East).
    Returns a tuple (best_energy, best_turns, n_examined):

    Easy: iterate over all 3^(n-2) turn lists
    Challenge: recursive depth-first search that places one residue at a time and abandons a branch the moment it hits an occupied point.
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Task 6: greedy heuristic
# ---------------------------------------------------------------------------
def greedy_search(sequence, rng=random):
    """Build ONE conformation greedily.

    For each new residue k = 2, 3, ..., n-1:
      * work out which of F/L/R lead to an unoccupied point;
      * for each of those, count the new H-H contacts residue k would make
      * choose uniformly at random among the turns making the most H-H contacts.
    If no turn is possible the chain is trapped: restart from scratch.

    Returns (turns, energy, restarts), where restarts is the number of times the chain got trapped before a complete fold was built.
    """
    # TODO
    raise NotImplementedError

# ---------------------------------------------------------------------------
# Task 7: simulated annealing
# ---------------------------------------------------------------------------
def simulated_annealing(sequence, n_iter=100000, T0=2.0, alpha=0.9995,
                        T_min=0.1, patience=None, rng=random,
                        history=None, history_every=100):
    """Simulated annealing over the turns list.

    a. Start from turns = ['F'] * (n - 2) and T = T0.
    b. Pick a random index i and a random turn d. turns' is a copy of turns with turns'[i] = d.
    c. Accept if E(turns') <= E(turns), otherwise with probability exp((E(turns) - E(turns')) / T). 
        Self-intersecting proposals (E = inf) are always rejected.
    d. T = max(T * alpha, T_min).
    e. Repeat for n_iter iterations, or stop early when the best energy has not improved for `patience` iterations (if patience is not None).

    Keep track of the best conformation ever visited, not just the current one.

    Returns (best_energy, best_turns).

    Extension (b): if history is a list, append the tuple
        (iteration, current_energy, best_energy, T, "".join(turns))
    at iteration 0 and every `history_every` iterations. Then
        python hp_animation.py --module hp_folding
    animates YOUR annealing run.

    Something to investigate: how many iterations does alpha = 0.9995 take
    to cool from T0 = 2.0 to 0.1? What happens if you choose alpha so that
    the cooling takes the whole run instead?
    """
    # TODO
    raise NotImplementedError

# ---------------------------------------------------------------------------
# Provided: draw a fold as ASCII art (H = hydrophobic, p = polar)
# ---------------------------------------------------------------------------
def draw(sequence, turns):
    coords = generate_coords(turns)
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    x0, y1 = min(xs), max(ys)
    W, H = 2 * (max(xs) - x0) + 1, 2 * (y1 - min(ys)) + 1
    grid = [[" "] * W for _ in range(H)]
    for i, (x, y) in enumerate(coords):
        gx, gy = 2 * (x - x0), 2 * (y1 - y)
        grid[gy][gx] = "H" if sequence[i] == "H" else "p"
        if i:
            px, py = coords[i - 1]
            grid[(gy + 2 * (y1 - py)) // 2][(gx + 2 * (px - x0)) // 2] = \
                "-" if py == y else "|"
    return "\n".join("".join(r).rstrip() for r in grid)


# ---------------------------------------------------------------------------
# Driver: Tasks 5, 6, 8 and 9
# ---------------------------------------------------------------------------
LECTURE_SEQ = "HPHPPHHPHHPPHPHH"

BENCHMARKS = [  # Unger & Moult (1993), optimal 2D energies
    ("HPHPPHHPHPPHPHHPPHPH", -9),
    ("HHPPHPPHPPHPPHPPHPPHPPHH", -9),
    ("PPHPPHHPPPPHHPPPPHHPPPPHH", -8),
]


def section(title, fn):
    print("\n" + title)
    try:
        fn()
    except NotImplementedError:
        print("  (skipped: not implemented yet)")


def task_1_to_3():
    c = generate_coords("LL")
    print("  HPPH, turns LL ->", c, "valid:", is_valid_walk(c),
          "E =", energy("HPPH", c))


def task_5():
    print(f"  {'n':>3} {'prefix':<16} {'#SAWs':>8} {'best E':>6} "
          f"{'#optimal':>8} {'time/s':>7}")
    for n in (6, 8, 10, 12, 14):
        s = LECTURE_SEQ[:n]
        t0 = time.perf_counter()
        e, folds, count = enumerate_conformations(s)
        print(f"  {n:>3} {s:<16} {count:>8} {e:>6} {len(folds):>8} "
              f"{time.perf_counter() - t0:>7.2f}")


def task_6():
    for n in (10, 14):
        s = LECTURE_SEQ[:n]
        e, t, r = greedy_search(s, 10000)
        print(f"  {s}: best E = {e}, total restarts = {r}")


def task_8_9():
    for s, opt in BENCHMARKS:
        g_e, _, _ = greedy_search(s, 10000)
        sa_e, sa_t = simulated_annealing(s, n_iter=100000)
        print(f"  n={len(s)}: optimum {opt}, greedy {g_e}, SA {sa_e}")
        print(draw(s, sa_t))


if __name__ == "__main__":
    random.seed(1)
    section("Tasks 1-3: coordinates, validity, energy", task_1_to_3)
    section("Task 5: brute force on prefixes of " + LECTURE_SEQ, task_5)
    section("Task 6: greedy search", task_6)
    section("Tasks 8-9: greedy vs simulated annealing on benchmarks", task_8_9)
