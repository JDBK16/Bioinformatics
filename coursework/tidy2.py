import random
import math
import time
import statistics
import csv
import json
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

rotate = {
    (1, 0): {"L": (0, -1), "F": (1, 0), "R": (0, 1)},
    (0, 1): {"L": (1, 0), "F": (0, 1), "R": (-1, 0)},
    (-1, 0): {"L": (0, 1), "F": (-1, 0), "R": (0, -1)},
    (0, -1): {"L": (-1, 0), "F": (0, -1), "R": (1, 0)}
}

best_energy = 0
best_turns = []
energy_evaluations = 0
trace = []
start_time = 0

#-------------------#
#----- Helper ------#
#-------------------#


# Compute the fraction of progress through the function for use in temperature()
def calc_progress(steps_done, max_steps, start_time, time_limit):
    return max(steps_done / max_steps, (time.perf_counter() - start_time) / time_limit)


# Calculate T based on progress through the run of the function rather than using a constant alpha value
# This ensures T follows a similar curve for both the Genetic Algorithm and Simulated Annealing Algorithm to keep the comparisons fair
def temperature(T0, T_min, progress):
    return T0 * (T_min / T0) ** min(progress, 1.0)

    

# Convert string of turns into array of coordinates
def generate_coords(turns):
    coords = [(0, 0), (1, 0)]
    direction = (1, 0)
    
    for t in turns:
        direction = rotate[direction][t]
        position = (coords[-1][0] + direction[0], coords[-1][1] + direction[1])
        coords.append(position)
        
    return coords


# Compute the energy of a HP-folding
def calculate_energy(sequence, coords):
    global energy_evaluations
    energy_evaluations += 1
    
    energy = 0
    positions = {coord: i for i, coord in enumerate(coords)}
    
    for i, (x, y) in enumerate(coords):
        if sequence[i] != "H":
            continue

        neighbours = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        
        for neighbour in neighbours:
            if neighbour in positions:
                j = positions[neighbour]
                if j > i and abs(i - j) > 1 and sequence[j] == "H":
                    energy -= 1
                    
    return energy


# If every coordinate is distinct then the walk is self avoiding
def is_valid_walk(coords):
    return len(set(coords)) == len(coords)


def update_best(turns, energy):
    global best_turns, best_energy
    trace.append([turns, energy, time.perf_counter() - start_time, energy_evaluations])
    best_turns, best_energy = turns, energy


#----------------------#
#----- Crossover ------#
#----------------------#


# Compute the probability of choosing any member of the population for crossover
# p[i] = energy(i) / (sum(energy(j)) for j in population)
# Turn this into a cumulative list for use by random
# https://doi.org/10.1006/jmbi.1993.1258 - "Genetic Algorithms for Protein Folding Simulations"
def crossover_probabilities(energies):
    total = sum(energies)
    
    if total == 0:
        return [(i + 1)/len(energies) for i in range(len(energies))]
    
    probs = [energies[0] / total]
    for i in range(1, len(energies)):
        probs.append(probs[-1] + (energies[i] / total))
        
    return probs


# Crossover two sets of turns - randomly choose a point to split two turns (t1, t2) and append t1[:split] to t2[split:]
# Try all turns to connect them in a random order - select the first that creates a self avoiding path
# Add the crossover to the new population if energy(crossover) <= (energy(t1) + energy(t2)) / 2 OR if a random float (0, 1) < e^((e_avg - e_cross) / T)
# https://doi.org/10.1006/jmbi.1993.1258 - "Genetic Algorithms for Protein Folding Simulations"
def crossover(sequence, t1, t2, e1, e2, T):
    split = random.randrange(1, len(t1))
    cross = t1[:split] + t2[split:]
    turn = random.sample(["L", "F", "R"], 3)
    valid = False
    for t in turn:
        cross[split] = t
        coords = generate_coords(cross)
        if is_valid_walk(coords): 
            valid = True
            break
    if valid == False:
        return None, None
    e_avg = (e1 + e2) / 2
    e3 = calculate_energy(sequence, coords)
    if e3 <= e_avg or random.random() < math.exp((e_avg - e3) / T):
        return cross, e3
    return None, None
        

#---------------------#
#----- Mutation ------#
#---------------------#


# Randomly select a turn from turns and replace it with a random turn
# If the mutated turns has a lower energy, or random (0, 1) < e^((energy(turns) - energy(mutation)) / T), return the mutation
# Otherwise return original turns
# https://doi.org/10.1006/jmbi.1993.1258 - "Genetic Algorithms for Protein Folding Simulations"
def mutate(sequence, turns, energy, T, mutation_steps):
    global best_turns, best_energy, trace, start_time
    for _ in range(mutation_steps):
        e1 = energy
        i = random.randrange(len(turns))
        mutant = turns.copy()
        mutant[i] = random.choice([x for x in "LFR" if x != turns[i]])
        mutant_coords = generate_coords(mutant)
        if is_valid_walk(mutant_coords):
            e2 = calculate_energy(sequence, mutant_coords)
            if e2 <= e1 or random.random() < math.exp((e1 - e2) / T):
                turns, energy = mutant, e2
                if energy < best_energy:
                    update_best(turns, energy)
                
    return turns, energy


#--------------------------------#
#-----  Genetic Algorithm  ------#
#--------------------------------#
    

def genetic_algorithm(sequence, population_size, mutation_steps, T, T_min, max_energy_evaluations = math.inf, time_limit = math.inf):
    global best_energy, best_turns, energy_evaluations, trace, start_time
    start_time = time.perf_counter()
    best_energy, energy_evaluations, best_turns = 0, 0, ["F"] * (len(sequence) - 2)
    trace = trace = [[["F"] * (len(sequence) - 2), 0, 0, 0]]
    
    T0 = T

    population = [["F"] * (len(sequence) - 2) for _ in range(population_size)]
    energies = [0] * population_size
    
    while energy_evaluations < max_energy_evaluations and ((time.perf_counter() - start_time) < time_limit):
        mutants, mutant_energies = [], []
        best_mutant, best_mutant_energy = [], math.inf
        for p in range(len(population)):
            mutant, mutant_energy = mutate(sequence, population[p], energies[p], T, mutation_steps)
            mutants.append(mutant)
            mutant_energies.append(mutant_energy)
            if mutant_energy < best_mutant_energy:
                best_mutant, best_mutant_energy = mutant, mutant_energy
                if mutant_energy < best_energy:
                    update_best(mutant, mutant_energy)
            T = temperature(T0, T_min, calc_progress(energy_evaluations, max_energy_evaluations, start_time, time_limit))
            if energy_evaluations >= max_energy_evaluations or (time.perf_counter() - start_time) >= time_limit:
                break
                    
        new_population = [best_mutant] # elitism
        new_energies = [best_mutant_energy]    
        cross_probs = crossover_probabilities(mutant_energies)
            
        while len(new_population) < population_size and energy_evaluations < max_energy_evaluations and ((time.perf_counter() - start_time) < time_limit):
            i_1, i_2 = random.choices(range(population_size), cum_weights=cross_probs, k=2)  
            new_turns, new_energy = crossover(sequence, mutants[i_1], mutants[i_2], mutant_energies[i_1], mutant_energies[i_2], T)
            if new_turns is not None:
                new_population.append(new_turns)
                new_energies.append(new_energy)
                if new_energy < best_energy:
                    update_best(new_turns, new_energy)
            
            
            T = temperature(T0, T_min, calc_progress(energy_evaluations, max_energy_evaluations, start_time, time_limit))
                        
        population = new_population
        energies = new_energies
    return best_turns, best_energy


#----------------------------------#
#-----  Simulated Annealing  ------#
#----------------------------------#


def simulated_annealing(sequence, T, T_min, max_energy_evaluations = math.inf, time_limit = math.inf):
    global best_energy, best_turns, energy_evaluations, trace, start_time
    start_time = time.perf_counter()
    best_energy, energy_evaluations, best_turns = 0, 0, ["F"] * (len(sequence) - 2)
    trace = [[["F"] * (len(sequence) - 2), 0, 0, 0]]
    T0 = T
    
    turns = ["F"]*(len(sequence) - 2)
    current_energy = 0
    
    while energy_evaluations < max_energy_evaluations and ((time.perf_counter() - start_time) < time_limit):
    
        i = random.randrange(0, len(turns))
        t = random.choice([turn for turn in "LFR" if turn != turns[i]])
        new_turns = turns.copy()
        new_turns[i] = t
        new_coords = generate_coords(new_turns)
        if is_valid_walk(new_coords):
            new_energy = calculate_energy(sequence, new_coords)
            if new_energy <= current_energy or (random.random() < math.exp((current_energy - new_energy) / T)):
                current_energy = new_energy
                turns = new_turns
                if new_energy < best_energy:
                    update_best(new_turns, new_energy)

        T = temperature(T0, T_min, calc_progress(energy_evaluations, max_energy_evaluations, start_time, time_limit))
    return best_turns, best_energy



#-------------------#
#-----  Data  ------#
#-------------------#

# https://doi.org/10.1006/jmbi.1993.1258 - "Genetic Algorithms for Protein Folding Simulations"
paper_results = [
    {"name": "UM20", "sequence": "HPHPPHHPHPPHPHHPPHPH",                                             "optimum": -9,  "paper_results": (-9, 30492)},
    {"name": "UM24", "sequence": "HHPPHPPHPPHPPHPPHPPHPPHH",                                         "optimum": -9,  "paper_results": (-9, 30491)},
    {"name": "UM25", "sequence": "PPHPPHHPPPPHHPPPPHHPPPPHH",                                        "optimum": -8,  "paper_results": (-8, 20400)},
    {"name": "UM36", "sequence": "PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP",                             "optimum": -14, "paper_results": (-14, 301339)},
    {"name": "UM48", "sequence": "PPHPPHHPPHHPPPPPHHHHHHHHHHPPPPPPHHPPHHPPHPPHHHHH",                 "optimum": -22, "paper_results": (-22, 126547)},
    {"name": "UM50", "sequence": "HHPHPHPHPHHHHPHPPPHPPPHPPPPHPPPHPPPHPHHHHPHPHPHPHH",               "optimum": -21, "paper_results": (-21, 592887)},
    {"name": "UM60", "sequence": "PPHHHPHHHHHHHHPPPHHHHHHHHHHPHPPPHHHHHHHHHHHHPPPPHHHHHHPHHPHP",     "optimum": -34, "paper_results": (-34, 208781)},
    {"name": "UM64", "sequence": "HHHHHHHHHHHHPHPHPPHHPPHHPPHPPHHPPHHPPHPPHHPPHHPPHPHPHHHHHHHHHHHH", "optimum": -42, "paper_results": (-37, 187393)},
]


def random_sequence(length, rng, h_ratio):
    sequence = ""
    for _ in range(length):
        if rng.random() < h_ratio:
            sequence += "H"
        else:
            sequence += "P"
    return sequence


def brute_force(sequence):
    positions = {(0, 0): 0, (1, 0): 1}
    path = [(0, 0), (1, 0)]
    turns = []
    best, best_set, count = 0, [], 0

    def extend(direction, energy):
        nonlocal best, best_set, count
        if len(path) == len(sequence):
            count += 1
            if energy < best:
                best, best_set = energy, [turns.copy()]
            elif energy == best:
                best_set.append(turns.copy())
            return

        x, y = path[-1]
        for t in "LFR":
            d = rotate[direction][t]
            p = (x + d[0], y + d[1])
            if p in positions:
                continue

            energy_change = 0
            if sequence[len(path)] == "H":
                for q in ((p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)):
                    j = positions.get(q)
                    if j is not None and j < len(path) - 1 and sequence[j] == "H":
                        energy_change -= 1

            positions[p] = len(path)
            path.append(p)
            turns.append(t)
            
            extend(d, energy + energy_change)
            
            del positions[p]
            path.pop()
            turns.pop()

    extend((1, 0), 0)
    return best, best_set, count


def make_datasets(seed=2026):
    rng = random.Random(seed)
    h_ratio = 0.5
    short = []
    for n in range(10, 15):
        k = 0
        while k < 3:
            sequence = random_sequence(n, rng, h_ratio)
            optimal_energy, optimal_set, n_folds = brute_force(sequence)
            if len(optimal_set) / n_folds < 0.01:
                short.append({"name": "short" + str(n) + "_" + str(k), "sequence": sequence, "optimum": optimal_energy,
                              "n_folds": n_folds, "n_optimal": len(optimal_set)})
                k += 1

    long = []
    for n in range(20, 101, 20):
        for k in range(3):
            long.append({"name": "long" + str(n) + "_" + str(k), "sequence": random_sequence(n, rng, h_ratio), "optimum": None})

    return {"paper": paper_results, "short": short, "long": long}


#-------------------------#
#-----  Evaluation  ------#
#-------------------------#

# Every run is one dictionary: name, sequence, algorithm, seed, budget_type, budget, turns, energy, energy_evaluations, time, trace
# A budget is ("evaluations", number) or ("time", seconds)


# Run GA and SA on every sequence, for every budget and seed, and save all runs to one JSON file
def run_all(dataset, seeds, budgets, filename):
    runs = []
    for entry in dataset:
        sequence = entry["sequence"]
        for budget_type, budget in budgets:
            # Only one limit is used, the other is left at infinity
            max_evals = budget if budget_type == "evaluations" else math.inf
            time_limit = budget if budget_type == "time" else math.inf
            for seed in seeds:
                for algorithm in ("GA", "SA"):
                    random.seed(seed)
                    if algorithm == "GA":
                        turns, energy = genetic_algorithm(sequence, 200, len(sequence), 2, 0.1, max_evals, time_limit)
                    else:
                        turns, energy = simulated_annealing(sequence, 2, 0.1, max_evals, time_limit)
                    runs.append({"name": entry["name"], "sequence": sequence, "algorithm": algorithm, "seed": seed,
                                 "budget_type": budget_type, "budget": budget, "turns": turns, "energy": energy,
                                 "energy_evaluations": energy_evaluations, "time": time.perf_counter() - start_time, "trace": trace})
        print("done", entry["name"])
    with open(filename, "w") as f:
        json.dump(runs, f)


# For each run of one algorithm on one sequence: the evaluations it needed to reach target (inf if it never did)
def evals_to_target(runs, name, algorithm, target):
    evals = []
    for run in runs:
        if run["name"] == name and run["algorithm"] == algorithm:
            evals.append(math.inf)
            for _, energy, _, evaluations in run["trace"]:
                if energy <= target:
                    evals[-1] = evaluations
                    break
    return evals


#--------------------------------#
#-----  Fig 1. Folds VS n  ------#
#--------------------------------#

def count_folds(lengths):
    counts = {}
    for n in lengths:
        _, _, count = brute_force("P" * n)
        counts[n] = count
    return counts
    
    
def plot_fold_counts(fold_counts, filename):
    lengths = sorted(fold_counts)
    
    # Image creation code generated by claude.ai then modified
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(lengths, [fold_counts[n] for n in lengths], "o-", color="#257de0", linewidth=2, markersize=8)
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length n", fontsize=14)
    ax.set_ylabel("Number of valid folds", fontsize=14)
    ax.grid(True, color="#e7e7e7")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    
    

#--------------------------------------------------#
#-----  Table 1. Correctness vs Brute Force  ------#
#--------------------------------------------------#

def brute_force_table(runsfile, dataset, filename = "brute_force_table.csv"):
    with open(runsfile) as f:
        runs = json.load(f)

    rows = [["name", "length", "optimum", "optimal_folds", "GA_hits", "SA_hits"]]
    totals = {"GA": [0, 0], "SA": [0, 0]}  # [hits, runs]

    for entry in dataset:
        row = [entry["name"], len(entry["sequence"]), entry["optimum"], str(entry["n_optimal"]) + "/" + str(entry["n_folds"])]
        for algorithm in ("GA", "SA"):
            evals = evals_to_target(runs, entry["name"], algorithm, entry["optimum"])
            hits = sum(1 for e in evals if e != math.inf)
            row.append(str(hits) + "/" + str(len(evals)))
            totals[algorithm][0] += hits
            totals[algorithm][1] += len(evals)
        rows.append(row)
    rows.append(["Total", "", "", "", str(totals["GA"][0]) + "/" + str(totals["GA"][1]), str(totals["SA"][0]) + "/" + str(totals["SA"][1])])

    # Written by claude.ai
    with open(filename, "w", newline = "") as f:
        csv.writer(f).writerows(rows)


#-----------------------------------------------------#
#-----  Table 2. Comparison With Unger & Moult  ------#
#-----------------------------------------------------#

def paper_table(runsfile, dataset, filename = "paper_table.csv"):
    with open(runsfile) as f:
        runs = json.load(f)

    rows = [["name", "length", "optimum", "paper_energy", "paper_evals",
             "GA_best", "GA_hits", "GA_q1", "GA_median_evals", "GA_q3",
             "SA_best", "SA_hits", "SA_q1", "SA_median_evals", "SA_q3"]]

    for entry in dataset:
        paper_energy, paper_evals = entry["paper_results"]
        row = [entry["name"], len(entry["sequence"]), entry["optimum"], paper_energy, paper_evals]
        for algorithm in ("GA", "SA"):
            evals = evals_to_target(runs, entry["name"], algorithm, paper_energy)
            hits = sum(1 for e in evals if e != math.inf)
            best = min(run["energy"] for run in runs if run["name"] == entry["name"] and run["algorithm"] == algorithm)
            # Quartiles only make sense when more than half the runs reached the target
            if hits > len(evals) / 2:
                quartiles = statistics.quantiles(evals, n=4, method="inclusive")
            else:
                quartiles = ["-", "-", "-"]
            row += [best, str(hits) + "/" + str(len(evals))] + quartiles
        rows.append(row)

    with open(filename, "w", newline="") as f:
        csv.writer(f).writerows(rows)


#-----------------------------------------------#
#-----  Fig 2. Best Energy With Budgets   ------#
#-----------------------------------------------#

def plot_budget_runs(files, filename):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colours = {"GA": "#257de0", "SA": "#f47230"}
    line_styles = ["-", "--"]

    for i, runs_file in enumerate(files):
        with open(runs_file) as f:
            runs = json.load(f)
        name = runs[0]["name"]
        best_found = min(run["energy"] for run in runs)

        for ax, budget_type in ((axes[0], "evaluations"), (axes[1], "time")):
            budgets = sorted({run["budget"] for run in runs if run["budget_type"] == budget_type})

            for algorithm in ("GA", "SA"):
                medians, q1s, q3s = [], [], []
                for budget in budgets:
                    values = [run["energy"] / best_found for run in runs if run["budget_type"] == budget_type and run["budget"] == budget and run["algorithm"] == algorithm]
                    q1, median, q3 = statistics.quantiles(values, n=4, method="inclusive")
                    medians.append(median)
                    q1s.append(q1)
                    q3s.append(q3)

                ax.plot(budgets, medians, line_styles[i], color=colours[algorithm],
                        linewidth=2, marker="o", markersize=8, label=algorithm + ", " + name)
                ax.fill_between(budgets, q1s, q3s, color=colours[algorithm], alpha=0.15, linewidth=0)

            ax.set_xscale("log")
            ax.minorticks_off()
            ax.set_xticks(budgets)
            ax.set_xticklabels([f"{b:g}" for b in budgets])

    for ax, label in ((axes[0], "Energy evaluation budget"), (axes[1], "Time budget (s)")):
        ax.set_xlabel(label, fontsize=14)
        ax.tick_params(labelsize=12)
        ax.grid(True, color="#e7e7e7")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Median energy / best found", fontsize=14)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(filename, dpi=300)
    

#---------------------------------#
#-----  Fig 3. Draw Folds   ------#
#---------------------------------#


# Fold-drawing style adapted from the module's Workshop 6 animation code (hp_animation.py)
H_COLOUR = "#c8402f"
P_COLOUR = "#ffffff"
EDGE = "#333333"
CHAIN = "#555555"


# Line segments joining every pair of non-consecutive, adjacent H residues (each pair once)
def contact_segments(sequence, coords):
    positions = {c: i for i, c in enumerate(coords)}
    segments = []
    for i, (x, y) in enumerate(coords):
        if sequence[i] != "H":
            continue
        for neighbour in ((x + 1, y), (x, y + 1)):
            j = positions.get(neighbour)
            if j is not None and abs(i - j) > 1 and sequence[j] == "H":
                segments.append([(x, y), neighbour])
    return segments


# Draw one fold on a square window of the lattice, with beads and bonds sized to one lattice cell
def draw_fold(ax, fig, sequence, turns, title):
    coords = generate_coords(turns)
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]

    # Square window around the fold, one cell of space on each side
    span = max(max(xs) - min(xs), max(ys) - min(ys)) + 2
    centre_x = (max(xs) + min(xs)) / 2
    centre_y = (max(ys) + min(ys)) / 2
    ax.set_xlim(centre_x - span / 2, centre_x + span / 2)
    ax.set_ylim(centre_y - span / 2, centre_y + span / 2)
    ax.set_aspect("equal")

    # Lattice grid lines at every integer position
    ax.set_xticks(range(int(centre_x - span / 2), int(centre_x + span / 2) + 1))
    ax.set_yticks(range(int(centre_y - span / 2), int(centre_y + span / 2) + 1))
    ax.tick_params(length=0, labelbottom=False, labelleft=False)
    ax.grid(color="#e2e2e2", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ax.spines.values():
        side.set_color("#bbbbbb")

    # Size beads and bonds relative to one lattice cell (in points)
    box = ax.get_position()
    side_inches = min(box.width * fig.get_figwidth(), box.height * fig.get_figheight())
    cell = side_inches * 72 / span
    bead = (0.55 * cell) ** 2
    edge_width = max(0.6, 0.06 * cell)

    segments = contact_segments(sequence, coords)
    ax.add_collection(LineCollection(segments, colors=H_COLOUR, linestyles=(0, (2, 1.5)),
                                     linewidths=max(1.0, 0.1 * cell), zorder=1))
    ax.plot(xs, ys, "-", color=CHAIN, linewidth=max(1.0, 0.16 * cell), solid_capstyle="round", zorder=2)
    h_coords = [c for c, r in zip(coords, sequence) if r == "H"]
    p_coords = [c for c, r in zip(coords, sequence) if r == "P"]
    if h_coords:
        ax.scatter([c[0] for c in h_coords], [c[1] for c in h_coords], s=bead, c=H_COLOUR, edgecolors=EDGE, linewidths=edge_width, zorder=3)
    if p_coords:
        ax.scatter([c[0] for c in p_coords], [c[1] for c in p_coords], s=bead, c=P_COLOUR, edgecolors=EDGE, linewidths=edge_width, zorder=3)
    ax.scatter(xs[0], ys[0], s=bead * 2.2, facecolors="none", edgecolors="#222222", linewidths=edge_width * 1.5, zorder=4)

    ax.set_title(title, fontsize=14)
    ax.text(0.03, 0.97, "contacts = " + str(len(segments)), transform=ax.transAxes, va="top", fontsize=11,
            bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.9))


# Draw the lowest-energy fold found (by either algorithm) for each named sequence, side by side
def plot_best_folds(runs_file, names, filename):
    with open(runs_file) as f:
        runs = json.load(f)
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5.6))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.12, top=0.9, wspace=0.08)

    for ax, name in zip(axes, names):
        sequence_runs = [run for run in runs if run["name"] == name]
        best = min(sequence_runs, key=lambda run: run["energy"])
        draw_fold(ax, fig, best["sequence"], best["turns"], name + ": E = " + str(best["energy"]) + " (" + best["algorithm"] + ")")

    # One legend for all panels, below them
    handles = [plt.scatter([], [], s=80, c=H_COLOUR, edgecolors=EDGE, label="H (hydrophobic)"),
               plt.scatter([], [], s=80, c=P_COLOUR, edgecolors=EDGE, label="P (polar)"),
               plt.Line2D([], [], linestyle="--", color=H_COLOUR, label="H–H contact"),
               plt.scatter([], [], s=160, facecolors="none", edgecolors="#222222", label="Residue 0")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=12)
    fig.savefig(filename, dpi=300)


#-------------------#
#-----  Main  ------#
#-------------------#


if __name__ == "__main__":
    quick = False  # True for a quick test with 3 seeds instead of 20
    seeds = range(3) if quick else range(20)
    datasets = make_datasets()

    #--------------------------------#
    #-----  Fig 1. Folds VS n  ------#
    #--------------------------------#

    #plot_fold_counts(count_folds(range(4, 17)), "figures/fig1_fold_counts.png")

    #-----------------------------------------------#
    #-----  Fig 2. Best Energy With Budgets   ------#
    #-----------------------------------------------#

    budgets = [("evaluations", b) for b in [1000, 3000, 10000, 30000, 100000, 300000]] + [("time", b) for b in [0.1, 0.3, 1, 3, 10, 30]]
    run_all([datasets["paper"][7]], seeds, budgets, "budget_paper.json")
    run_all([datasets["long"][7]], seeds, budgets, "budget_long.json")
    plot_budget_runs(["budget_paper.json", "budget_long.json"], "figures/fig2_budgets.png")

    #---------------------------------#
    #-----  Fig 3. Draw Folds   ------#
    #---------------------------------#

    plot_best_folds("paper_runs.json", ["UM20", "UM48", "UM64"], "figures/fig4_folds.png")

    #--------------------------------------------------#
    #-----  Table 1. Correctness vs Brute Force  ------#
    #--------------------------------------------------#

    #run_all(datasets["short"], seeds, [("evaluations", 20000)], "short_runs.json")
    #brute_force_table("short_runs.json", datasets["short"])

    #-----------------------------------------------------#
    #-----  Table 2. Comparison With Unger & Moult  ------#
    #-----------------------------------------------------#

    #run_all(datasets["paper"], seeds, [("evaluations", 500000)], "paper_runs.json")
    #paper_table("paper_runs.json", datasets["paper"])
