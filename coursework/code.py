import random
import math
import time
import statistics
import csv
import json
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Given a direction and a turn this dictionary produces the new direction
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


# Compute the energy of a fold
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


# Check that a set of coordinates is self-avoiding
def is_valid_walk(coords):
    # A set contains only distinct values - if every coordinate is distinct then the walk is self-avoiding
    return len(set(coords)) == len(coords)


# Update the best turns and energy seen by an algorithm so far and append it to the algorithm trace (a record of the time/energy evaluations for each new best fold)
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
    
    # If all energies are 0 follow a uniform distribution
    if total == 0:
        return [(i + 1)/len(energies) for i in range(len(energies))]
    
    probs = [energies[0] / total]
    for i in range(1, len(energies)):
        probs.append(probs[-1] + (energies[i] / total))
        
    return probs


# Crossover two sets of turns - randomly choose a point to split two turns (t1, t2) and append t1[:split] to t2[split:]
# Try to connect them with turns in a random order - select the first that creates a self avoiding path
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
    global best_energy
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
    

# https://doi.org/10.1006/jmbi.1993.1258 - "Genetic Algorithms for Protein Folding Simulations"
def genetic_algorithm(sequence, population_size, mutation_steps, T, T_min, max_energy_evaluations = math.inf, time_limit = math.inf):
    global best_energy, best_turns, energy_evaluations, trace, start_time
    start_time = time.perf_counter()
    best_energy, energy_evaluations, best_turns = 0, 0, ["F"] * (len(sequence) - 2)
    trace = trace = [[["F"] * (len(sequence) - 2), 0, 0, 0]]
    T0 = T

    # Initialise the population to population_size * fully straight folds (all turns are "F") 
    population = [["F"] * (len(sequence) - 2) for _ in range(population_size)]
    # Initialise the energies of the population to all 0s
    energies = [0] * population_size
    
    # Continue the algorithm until the max_energy_evaluations has been reached or the time_limit has been reached (depending on which budget is selected)
    while energy_evaluations < max_energy_evaluations and ((time.perf_counter() - start_time) < time_limit):
        # Mutate every member of the population mutation_steps times
        mutants, mutant_energies = [], []
        best_mutant, best_mutant_energy = [], math.inf
        for p in range(len(population)):
            mutant, mutant_energy = mutate(sequence, population[p], energies[p], T, mutation_steps)
            mutants.append(mutant)
            mutant_energies.append(mutant_energy)
            # Update the mutated population's best and the total best turns
            if mutant_energy < best_mutant_energy:
                best_mutant, best_mutant_energy = mutant, mutant_energy
                if mutant_energy < best_energy:
                    update_best(mutant, mutant_energy)
            
            # Update temperature based on progress through budget (either energy evaluations or time)
            T = temperature(T0, T_min, calc_progress(energy_evaluations, max_energy_evaluations, start_time, time_limit))
            if energy_evaluations >= max_energy_evaluations or (time.perf_counter() - start_time) >= time_limit:
                break
        
        # Keep the best mutated turns in the new population without crossover
        new_population = [best_mutant]
        new_energies = [best_mutant_energy]    
        cross_probs = crossover_probabilities(mutant_energies)
        
        # Until the new population is filled or the budget has been reached, perform crossovers
        while len(new_population) < population_size and energy_evaluations < max_energy_evaluations and ((time.perf_counter() - start_time) < time_limit):
            # Select two members of the population with probability proportional to their energy
            i_1, i_2 = random.choices(range(population_size), cum_weights=cross_probs, k=2)  
            new_turns, new_energy = crossover(sequence, mutants[i_1], mutants[i_2], mutant_energies[i_1], mutant_energies[i_2], T)
            
            # If a valid fold is produced by the crossover add it to the new population and update the best turns
            if new_turns is not None:
                new_population.append(new_turns)
                new_energies.append(new_energy)
                if new_energy < best_energy:
                    update_best(new_turns, new_energy)
            
            # Update temperature based on progress through budget (either energy evaluations or time)
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
    
    # Initialise the turns to a straight protein with 0 energy (turns are all "F")
    turns = ["F"]*(len(sequence) - 2)
    current_energy = 0
    
    # While under the budget (either energy evaluations or time limit) continue the algorithm
    while energy_evaluations < max_energy_evaluations and ((time.perf_counter() - start_time) < time_limit):
        # Randomly select a turn
        i = random.randrange(0, len(turns))
        # Randomly swap it to one of the other turns
        t = random.choice([turn for turn in "LFR" if turn != turns[i]])
        new_turns = turns.copy()
        new_turns[i] = t
        
        # Confirm the swap results in a valid fold
        new_coords = generate_coords(new_turns)
        if is_valid_walk(new_coords):
            # Keep the change if the new fold's energy is less than the previous fold's OR with probability e^((current energy - new energy) / T)
            new_energy = calculate_energy(sequence, new_coords)
            if new_energy <= current_energy or (random.random() < math.exp((current_energy - new_energy) / T)):
                current_energy = new_energy
                turns = new_turns
                # Update the best turns
                if new_energy < best_energy:
                    update_best(new_turns, new_energy)

        # Update temperature based on progress through budget (either energy evaluations or time)
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


# Create a random sequence of Ps and Hs with ratio h_ratio of Hs
def random_sequence(length, rng, h_ratio):
    sequence = ""
    for _ in range(length):
        if rng.random() < h_ratio:
            sequence += "H"
        else:
            sequence += "P"
    return sequence


# Brute force a sequence to find all possible valid folds and therefore the optimum energy for the sequence
def brute_force(sequence):
    positions = {(0, 0): 0, (1, 0): 1}
    path = [(0, 0), (1, 0)]
    turns = []
    best, best_set, count = 0, [], 0

    def extend(direction, energy):
        nonlocal best, best_set, count
        
        # Base case - if the path reaches the correct length update the best values and return
        if len(path) == len(sequence):
            count += 1
            if energy < best:
                best, best_set = energy, [turns.copy()]
            elif energy == best:
                best_set.append(turns.copy())
            return

        # Get the last set of coords from the path
        x, y = path[-1]
        # Try each turn from these coords
        for t in "LFR":
            # Calculate the next coords with the selected turn
            d = rotate[direction][t]
            p = (x + d[0], y + d[1])
            # Check p doesn't already exist in the path (self avoiding)
            if p in positions:
                continue

            # Check if the new coordinate changes the energy
            energy_change = 0
            if sequence[len(path)] == "H":
                for q in ((p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)):
                    j = positions.get(q)
                    if j is not None and j < len(path) - 1 and sequence[j] == "H":
                        energy_change -= 1

            # Add new coords to path and positions, and the new turn to turns
            positions[p] = len(path)
            path.append(p)
            turns.append(t)
            
            extend(d, energy + energy_change)
            
            # Remove coords from path and positions and the new turn from turns to reset for the next t in the loop
            del positions[p]
            path.pop()
            turns.pop()

    extend((1, 0), 0)
    return best, best_set, count


# Create the datasets:
# 1. The data from the Unger & Moult paper
# 2. A set of short sequences (length 10-15) with brute-forced optimal values
# 3. A set of long sequences (lengths of multiples of 20 from 20 to 100) with no known optimal 
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

# 
def run_all(dataset, seeds, budgets, filename):
    # Skips running the algorithms if the data already exists
    if os.path.exists(filename):
        return
    
    runs = []
    # Iterate over each sequence in the dataset
    for entry in dataset:
        sequence = entry["sequence"]
        # Iterate over the given budgets (either max energy evaluations or time limit)
        for budget_type, budget in budgets:
            # Set the budget not being used to infinity to prevent early stop
            if budget_type == "time":
                max_evals, time_limit = math.inf, budget
            else:
                max_evals, time_limit = budget, math.inf
            
            # Iterate over each seed (the median is selected to avoid anomolous success/failure)    
            for seed in seeds:
                for algorithm in ("GA", "SA"):
                    random.seed(seed)
                    if algorithm == "GA":
                        turns, energy = genetic_algorithm(sequence, 200, len(sequence), 2, 0.1, max_evals, time_limit)
                    else:
                        turns, energy = simulated_annealing(sequence, 2, 0.1, max_evals, time_limit)
                        
                    # Track the result of each run
                    runs.append({"name": entry["name"], "sequence": sequence, "algorithm": algorithm, "seed": seed, "budget_type": budget_type, "budget": budget, "turns": turns, "energy": energy, "energy_evaluations": energy_evaluations, "time": time.perf_counter() - start_time, "trace": trace})
                    
        print("done", entry["name"])
        
    # Store it as json in a file to avoid rerunning
    with open(filename, "w") as f:
        json.dump(runs, f)


# Extract the json from a file
def load_runs(filename):
    with open(filename) as f:
        return json.load(f)


# Select certain runs by a certain algorithm
def select_algorithm_runs(runs, name, algorithm):
    return [run for run in runs if run["name"] == name and run["algorithm"] == algorithm]


# Written by claude.ai to write data to a CSV file
def write_csv(filename, rows):
    with open(filename, "w", newline="") as f:
        csv.writer(f).writerows(rows)


# Set an appearance for all figures
def style(ax, xlabel):
    ax.set_xlabel(xlabel, fontsize=14)
    ax.grid(True, color="#e7e7e7")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


#--------------------------------------------------#
#-----  Table 1. Correctness vs Brute Force  ------#
#--------------------------------------------------#


# Compare the best results of the SA and GA algorithms against the optimal results calculated by brute force
def brute_force_table(runsfile, dataset, filename = "brute_force_table.csv"):
    # Load the runs of SA and GA on the optimal-known short sequences
    runs = load_runs(runsfile)
    # Set column names
    rows = [["name", "length", "optimum", "optimal_folds", "GA_hits", "SA_hits"]]
    # [optimal hits, runs]
    total = {"GA": [0, 0], "SA": [0, 0]}
    
    # Iterate over each sequence in the dataset
    for entry in dataset:
        
        row = [entry["name"], len(entry["sequence"]), entry["optimum"], str(entry["n_optimal"]) + "/" + str(entry["n_folds"])]
        for algorithm in ("GA", "SA"):
            # Extract the energies from all runs of the algorithm on the sequence
            energies = [run["energy"] for run in select_algorithm_runs(runs, entry["name"], algorithm)]
            # Count the number of optimum energies
            hits = energies.count(entry["optimum"])
            row.append(str(hits) + "/" + str(len(energies)))
            total[algorithm][0] += hits
            total[algorithm][1] += len(energies)
        rows.append(row)
    
    # Add the total number of hits / runs for each algorithm
    rows.append(["Total", "", "", "", str(total["GA"][0]) + "/" + str(total["GA"][1]), str(total["SA"][0]) + "/" + str(total["SA"][1])])
    
    write_csv(filename, rows)


#-----------------------------------------------------#
#-----  Table 2. Comparison With Unger & Moult  ------#
#-----------------------------------------------------#


# Compare the Unger & Moult results to the GA and SA results on the sequences from the paper
def paper_table(runsfile, dataset, filename = "paper_table.csv"):
    runs = load_runs(runsfile)
    # Store the best result, number of optimum results, lower quartile, median, and upper quartile of SA and GA
    rows = [["name", "length", "optimum", "paper_energy", "paper_evals",
             "GA_best", "GA_hits", "GA_q1", "GA_median_evals", "GA_q3",
             "SA_best", "SA_hits", "SA_q1", "SA_median_evals", "SA_q3"]]
    
    # Iterate over each sequence in the dataset
    for entry in dataset:
        # Extract the papers's data about the sequence
        paper_energy, paper_evals = entry["paper_results"]
        row = [entry["name"], len(entry["sequence"]), entry["optimum"], paper_energy, paper_evals]
        
        # Try each algorithm
        for algorithm in ("GA", "SA"):
            # Select all runs on this sequence by this algorithm
            algorithm_runs = select_algorithm_runs(runs, entry["name"], algorithm)
            
            # Compute evaluations for each run to reach the paper's energy (inf if it didn't)
            evals = []
            for run in algorithm_runs:
                reached = [evaluations for _, energy, _, evaluations in run["trace"] if energy <= paper_energy]
                evals.append(reached[0] if reached else math.inf)
            
            # Number of runs that reached the optimum
            hits = sum(1 for e in evals if e != math.inf)
            
            # Quartiles only useful if over half of the runs reached the target
            if hits > len(evals) / 2:
                quartiles = statistics.quantiles(evals, n=4, method="inclusive")
            else: 
                quartiles = ["-", "-", "-"]
            
            # APpend best energy, number of optimum hits, ratio of hits to runs, and quartiles to the table
            row += [min(run["energy"] for run in algorithm_runs), str(hits) + "/" + str(len(evals))] + quartiles
            
        rows.append(row)
        
    write_csv(filename, rows)
    
    
#--------------------------------#
#-----  Fig 1. Folds VS n  ------#
#--------------------------------#


# Count valid folds for an all P sequence of each length by brute force to demonstrate how the number of turns scales with sequence length
def plot_fold_counts(lengths, filename):
    counts = [brute_force("P" * n)[2] for n in lengths]
    
    # Image creation code generated by claude.ai then modified
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(lengths, counts, "o-", color="#257de0", linewidth=2, markersize=8)
    ax.set_yscale("log")
    ax.set_ylabel("Number of valid folds", fontsize=14)
    style(ax, "Sequence length n")
    fig.tight_layout()
    fig.savefig(filename, dpi=300)


#-----------------------------------------------#
#-----  Fig 2. Best Energy With Budgets   ------#
#-----------------------------------------------#

# Median (line) and quartiles (shaded) of energy / best found at each budget, evaluations on the left and time on the right
# Code to create figures generated by claude.ai and modified to work with my functions
def plot_budget_runs(files, filename):
    # plt code created by claude.ai
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colours = {"GA": "#257de0", "SA": "#f47230"}
    
    for runs_file, line_style in zip(files, ["-", "--"]):
        # Load the runs
        runs = load_runs(runs_file)
        # Select the minimum energy from all runs
        best_found = min(run["energy"] for run in runs)
        # Run for both time budget and energy evaluations budget
        for ax, budget_type in ((axes[0], "evaluations"), (axes[1], "time")):
            # Sort budgets
            budgets = sorted({run["budget"] for run in runs if run["budget_type"] == budget_type})
            # Iterate over both algorithms
            for algorithm in ("GA", "SA"):
                q1s, medians, q3s = [], [], []
                # Iterate over each budget test
                for budget in budgets:
                    values = [run["energy"] / best_found for run in runs if run["algorithm"] == algorithm and run["budget_type"] == budget_type and run["budget"] == budget]
                    q1, median, q3 = statistics.quantiles(values, n=4, method="inclusive")
                    q1s.append(q1)
                    medians.append(median)
                    q3s.append(q3)
                    
                ax.plot(budgets, medians, line_style, color=colours[algorithm], linewidth=2, marker="o", markersize=8, label=algorithm + ", " + runs[0]["name"])
                ax.fill_between(budgets, q1s, q3s, color=colours[algorithm], alpha=0.15, linewidth=0)
            ax.set_xscale("log")
            ax.minorticks_off()
            ax.set_xticks(budgets)
            ax.set_xticklabels([f"{b:g}" for b in budgets])

    # Figure code created by claude.ai
    for ax, label in ((axes[0], "Energy evaluation budget"), (axes[1], "Time budget (s)")):
        style(ax, label)
        ax.tick_params(labelsize=12)
    axes[0].set_ylabel("Median energy / best found", fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(filename, dpi=300)


#---------------------------------#
#-----  Fig 3. Draw Folds   ------#
#---------------------------------#


# Fold drawing code adapted from the Workshop 5 animation code (hp_animation.py)
H_COLOUR = "#c8402f"
P_COLOUR = "#ffffff"
EDGE = "#333333"
CHAIN = "#555555"


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

    # Lattice grid lines at every integer
    ax.set_xticks(range(int(centre_x - span / 2), int(centre_x + span / 2) + 1))
    ax.set_yticks(range(int(centre_y - span / 2), int(centre_y + span / 2) + 1))
    ax.tick_params(length=0, labelbottom=False, labelleft=False)
    ax.grid(color="#e2e2e2", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ax.spines.values():
        side.set_color("#bbbbbb")

    # Size beads and bonds relative to one lattice cell
    box = ax.get_position()
    cell = min(box.width * fig.get_figwidth(), box.height * fig.get_figheight()) * 72 / span
    bead = (0.55 * cell) ** 2
    edge_width = max(0.6, 0.06 * cell)

    # Line segments joining every pair of non-consecutive, adjacent H residues
    positions = {c: i for i, c in enumerate(coords)}
    segments = []
    for i, (x, y) in enumerate(coords):
        for neighbour in ((x + 1, y), (x, y + 1)):
            j = positions.get(neighbour)
            if sequence[i] == "H" and j is not None and abs(i - j) > 1 and sequence[j] == "H":
                segments.append([(x, y), neighbour])
    ax.add_collection(LineCollection(segments, colors=H_COLOUR, linestyles=(0, (2, 1.5)), linewidths=max(1.0, 0.1 * cell), zorder=1))

    # Chain, then H and P beads, then a ring around residue 0
    ax.plot(xs, ys, "-", color=CHAIN, linewidth=max(1.0, 0.16 * cell), solid_capstyle="round", zorder=2)
    for residue, colour in (("H", H_COLOUR), ("P", P_COLOUR)):
        points = [c for c, r in zip(coords, sequence) if r == residue]
        if points:
            ax.scatter([c[0] for c in points], [c[1] for c in points], s=bead, c=colour, edgecolors=EDGE, linewidths=edge_width, zorder=3)
    ax.scatter(xs[0], ys[0], s=bead * 2.2, facecolors="none", edgecolors="#222222", linewidths=edge_width * 1.5, zorder=4)

    ax.set_title(title, fontsize=14)
    ax.text(0.03, 0.97, "contacts = " + str(len(segments)), transform=ax.transAxes, va="top", fontsize=11,
            bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.9))


# Draw the lowest-energy fold found by either algorithm for each named sequence, side by side
def plot_best_folds(runs_file, names, filename):
    runs = load_runs(runs_file)
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5.6))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.12, top=0.9, wspace=0.08)

    for ax, name in zip(axes, names):
        best = min([run for run in runs if run["name"] == name], key=lambda run: run["energy"])
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
    seeds = range(20)
    datasets = make_datasets()

    #--------------------------------#
    #-----  Fig 1. Folds VS n  ------#
    #--------------------------------#

    plot_fold_counts(range(4, 17), "figures/fig1_fold_counts.png")

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

    run_all(datasets["short"], seeds, [("evaluations", 20000)], "short_runs.json")
    brute_force_table("short_runs.json", datasets["short"])

    #-----------------------------------------------------#
    #-----  Table 2. Comparison With Unger & Moult  ------#
    #-----------------------------------------------------#

    run_all(datasets["paper"], seeds, [("evaluations", 500000)], "paper_runs.json")
    paper_table("paper_runs.json", datasets["paper"])
