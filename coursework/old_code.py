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


def evaluations_until_target(trace, target):
    for _, energy, _, evaluations in trace:
        if energy <= target:
            return evaluations
    return None


def evaluate(runs, target):
    energies = sorted(run["energy"] for run in runs)
    hits = [evaluations_until_target(run["trace"], target) for run in runs]
    successes = sorted(h for h in hits if h is not None)

    if successes:
        best_evals = successes[0]
    else:
        best_evals = None

    if len(successes) > len(runs) / 2:
        median_evals = statistics.median([math.inf if h is None else h for h in hits])
    else:
        median_evals = None

    return {"best_energy": energies[0], "median_energy": statistics.median(energies), "success": str(len(successes)) + "/" + str(len(runs)), "median_evals": median_evals, "best_evals": best_evals}


def run_algorithms(sequence, seeds, max_evaluations=math.inf, time_limit=math.inf, T0=2, T_min=0.1, population_size=200):
    runs = []
    for seed in seeds:
        random.seed(seed)
        turns, energy = genetic_algorithm(sequence, population_size, len(sequence), T0, T_min, max_energy_evaluations=max_evaluations, time_limit=time_limit)
        runs.append({"algorithm": "GA", "seed": seed, "turns": turns, "energy": energy, "trace": trace, "energy_evaluations": energy_evaluations, "time": time.perf_counter() - start_time})

        random.seed(seed)
        turns, energy = simulated_annealing(sequence, T0, T_min, max_energy_evaluations=max_evaluations, time_limit=time_limit)
        runs.append({"algorithm": "SA", "seed": seed, "turns": turns, "energy": energy, "trace": trace, "energy_evaluations": energy_evaluations, "time": time.perf_counter() - start_time})
    return runs

    
def results_table(dataset, seeds, filename, max_evaluations=math.inf, time_limit=math.inf):
    rows = []
    for entry in dataset:
        runs = run_algorithms(entry["sequence"], seeds, max_evaluations, time_limit)

        if entry["optimum"] is not None:
            target = entry["optimum"]
        else:
            target = min(run["energy"] for run in runs)

        for algorithm in ("GA", "SA"):
            row = {"name": entry["name"], "length": len(entry["sequence"]), "target": target, "algorithm": algorithm}
            to_evaluate = [run for run in runs if run["algorithm"] == algorithm]
            row.update(evaluate(to_evaluate, target))
            
            if "paper_results" in entry:
                row["paper_results_energy"], row["paper_results_evals"] = entry["paper_results"]
            rows.append(row)
            
    # Written by claude.ai to write data to a CSV file
    field = []
    for row in rows:
        field += [key for key in row if key not in field]
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def save_runs(dataset, seeds, filename, max_evaluations=math.inf, time_limit=math.inf):
    all_runs = []
    for entry in dataset:
        for run in run_algorithms(entry["sequence"], seeds, max_evaluations, time_limit):
            run["name"] = entry["name"]
            run["sequence"] = entry["sequence"]
            all_runs.append(run)
        print("done", entry["name"])
    with open(filename, "w") as f:
        json.dump(all_runs, f)
        
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
    
    rows = []
    totals = {"GA": [0, 0], "SA": [0, 0]}
    
    for entry in dataset:
        row = {"name": entry["name"], "length": len(entry["sequence"]), "optimum": entry["optimum"], "optimal_folds": str(entry["n_optimal"]) + "/" + str(entry["n_folds"])}
        
        for algorithm in ("GA", "SA"):
            algorithm_runs = [run for run in runs if run["name"] == entry["name"] and run["algorithm"] == algorithm]
            hits = sum(1 for run in algorithm_runs if run["energy"] == entry["optimum"])
            row[algorithm + "_hits"] = str(hits) + "/" + str(len(algorithm_runs))
            totals[algorithm][0] += hits
            totals[algorithm][1] += len(algorithm_runs)
        rows.append(row)
    
    rows.append({"name": "Total", "GA_hits": str(totals["GA"][0]) + "/" + str(totals["GA"][1])})
    
    # Written by claude.ai
    with open(filename, "w", newline = "") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "length", "optimum", "optimal_folds", "GA_hits", "SA_hits"])
        writer.writeheader()
        writer.writerows(rows)
        
        
#-----------------------------------------------------#
#-----  Table 2. Comparison With Unger & Moult  ------#
#-----------------------------------------------------#
    
def paper_table(runsfile, dataset, filename = "paper_table.csv"):
    with open(runsfile) as f:
        runs = json.load(f)
        
    rows = []
    for entry in dataset:
        paper_energy, paper_evals = entry["paper_results"]
        row = {"name": entry["name"], "length": len(entry["sequence"]), "optimum": entry["optimum"], "paper_energy": paper_energy, "paper_evals": paper_evals}
        
        evals = {}
        for algorithm in ("GA", "SA"):
            algorithm_runs = [run for run in runs if run["name"] == entry["name"] and run["algorithm"] == algorithm]
            hits = [evaluations_until_target(run["trace"], paper_energy) for run in algorithm_runs]
            successes = [h for h in hits if h is not None]
            evals[algorithm] = [math.inf if h is None else h for h in hits]
            
            row[algorithm + "_best"] = min(run["energy"] for run in algorithm_runs)
            row[algorithm + "_hits"] = str(len(successes)) + "/" + str(len(algorithm_runs))
            
            if len(successes) > len(algorithm_runs) / 2:
                q1, median, q3 = statistics.quantiles(evals[algorithm], n=4, method="inclusive")
                row[algorithm + "_median_evals"] = median
                row[algorithm + "_q1"] = q1
                row[algorithm + "_q3"] = q3
            else:
                row[algorithm + "_median_evals"] = "-"
                row[algorithm + "_q1"] = "-"
                row[algorithm + "_q3"] = "-"
    
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        
        
#-----------------------------------------------#
#-----  Fig 2. Best Energy With Budgets   ------#
#-----------------------------------------------#

def run_budgets(entry, seeds, filename, eval_budgets, time_budgets):
    all_runs = []
    for eval in eval_budgets:
        runs = run_algorithms(entry["sequence"], seeds, max_evaluations = eval)
        for run in runs:
            run["name"] = entry["name"]
            run["limit"] = "evaluations"
            run["budget"] = eval
            all_runs.append(run)
        print(entry["name"], "Evaluations", eval, "Done")
    
    for time in time_budgets:
        runs = run_algorithms(entry["sequence"], seeds, time_limit = time)
        for run in runs:
            run["name"] = entry["name"]
            run["limit"] = "time"
            run["budget"] = time
            all_runs.append(run)
        print(entry["name"], "Time", time, "Done")
    
    with open(filename, "w") as f:
        json.dump(all_runs, f)

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
    quick = False
    if quick == False:
        seeds = range(20)
    else:
        seeds = range(3)
    datasets = make_datasets()

    #--------------------------------#
    #-----  Fig 1. Folds VS n  ------#
    #--------------------------------#
    
    #fold_counts = count_folds(range(4, 17))
    #plot_fold_counts(fold_counts, "figures/fig1_fold_counts.png")
    
    
    #-----------------------------------------------#
    #-----  Fig 2. Best Energy With Budgets   ------#
    #-----------------------------------------------#
    
    eval_budgets = [1000, 3000, 10000, 30000, 100000, 300000]
    time_budgets = [0.1, 0.3, 1, 3, 10, 30]
    run_budgets(datasets["paper"][7], seeds, "budget_paper.json", eval_budgets, time_budgets)
    run_budgets(datasets["long"][7], seeds, "budget_long.json", eval_budgets, time_budgets)
    plot_budget_runs(["budget_paper.json", "budget_long.json"])
    
    
    #---------------------------------#
    #-----  Fig 3. Draw Folds   ------#
    #---------------------------------#
    
    plot_best_folds("paper_runs.json", ["UM20", "UM48", "UM64"], "figures/fig4_folds.png")
    
    #--------------------------------------------------#
    #-----  Table 1. Correctness vs Brute Force  ------#
    #--------------------------------------------------#
    
    #save_runs(datasets["short"], seeds, "short_runs.json", max_evaluations = 20000)
    #brute_force_table("short_runs.json", datasets["short"])
    
    
    #-----------------------------------------------------#
    #-----  Table 2. Comparison With Unger & Moult  ------#
    #-----------------------------------------------------#
    
    #save_runs(datasets["paper"], seeds, "paper_runs.json", max_evaluations = 500000)
    #paper_table("paper_runs.json", datasets["paper"])