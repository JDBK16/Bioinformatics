import random
import math
import time

rotate = {
    (1, 0): {"L": (0, -1), "F": (1, 0), "R": (0, 1)},
    (0, 1): {"L": (1, 0), "F": (0, 1), "R": (-1, 0)},
    (-1, 0): {"L": (0, 1), "F": (-1, 0), "R": (0, -1)},
    (0, -1): {"L": (-1, 0), "F": (0, -1), "R": (1, 0)}
}

#-------------------#
#----- Helper ------#
#-------------------#

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


#----------------------#
#----- Crossover ------#
#----------------------#


# Compute the probability of choosing any member of the population for crossover
# p[i] = energy(i) / (sum(energy(j)) for j in population)
# https://doi.org/10.1006/jmbi.1993.1258 - "Genetic Algorithms for Protein Folding Simulations"
def crossover_probabilities(energies):
    probs = []
    total = sum(energies)
    if total == 0:
        return [1/len(energies) for _ in energies]
    for energy in energies:
        probs.append(energy / total)
    return probs


# Crossover two sets of turns - randomly choose a point to split two turns (t1, t2) and append t1[:split] to t2[split:]
# Try all turns to connect them in a random order - select the first that creates a self avoiding path
# Add the crossover to the new population if energy(crossover) <= (energy(t1) + energy(t2)) / 2 OR if a random float (0, 1) < e^((e_avg - e_cross) / T)
# https://doi.org/10.1006/jmbi.1993.1258 - "Genetic Algorithms for Protein Folding Simulations"
def crossover(sequence, parent_1, parent_2, T):
    (t1, e1) = parent_1
    (t2, e2) = parent_2
    split = random.randrange(1, len(t1))
    cross = t1[:split] + t2[split:]
    options = ["L", "F", "R"]
    options.remove(cross[split])
    if not is_valid_walk(generate_coords(cross)): 
        valid = False
        while len(options) > 0:
            turn = random.choice(options)
            options.remove(turn)
            cross[split] = turn
            if is_valid_walk(generate_coords(cross)):
                valid = True
                break
        if not valid:
            return None, None
        
    e_avg = (e1 + e2) / 2
    e3 = calculate_energy(sequence, generate_coords(cross))
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
def mutate(sequence, turns, energy, T):
    e1 = energy
    i = random.randrange(len(turns))
    
    options = ["L", "F", "R"]
    options.remove(turns[i])
    
    mutation = turns.copy()
    while len(options) > 0:
        t = random.choice(options)
        options.remove(t)
        
        mutation[i] = t
        mutation_coords = generate_coords(mutation)
        
        if is_valid_walk(mutation_coords):
            e2 = calculate_energy(sequence, mutation_coords)
            if e2 <= e1 or random.random() < math.exp((e1 - e2) / T):
                return mutation, e2
    
    return turns, energy


#--------------------------------#
#-----  Genetic Algorithm  ------#
#--------------------------------#
    

def genetic_algorithm(sequence, population_size, T, alpha, T_min, generations = math.inf, time_limit = math.inf):
    start_time = time.time()
    
    population = [["F"] * (len(sequence) - 2)] * population_size
    energies = [0] * population_size

    best_energy = 0
    best_turns = population[0]
    
    gen = 0
    while gen < generations and ((time.time() - start_time) < time_limit):
        gen += 1
        new_population = [best_turns] # elitism
        new_energies = [best_energy]
        cross_probs = crossover_probabilities(energies)
        while len(new_population) < population_size and ((time.time() - start_time) < time_limit):
            
            i_1, i_2 = random.choices(range(population_size), weights=cross_probs, k=2)  
            mutation_1, mut_energy_1 = mutate(sequence, population[i_1], energies[i_1], T)
            mutation_2, mut_energy_2 = mutate(sequence, population[i_2], energies[i_2], T)
            
            cross, cross_energy = crossover(sequence, (mutation_1, mut_energy_1), (mutation_2, mut_energy_2), T)
            if cross is not None:
                new_population.append(cross)
                new_energies.append(cross_energy)
                if cross_energy <= best_energy:
                    best_energy = cross_energy
                    best_turns = cross
            else:
                if mut_energy_1 < mut_energy_2:
                    new_population.append(mutation_1)
                    new_energies.append(mut_energy_1)
                else:
                    new_population.append(mutation_2)
                    new_energies.append(mut_energy_2)
                        
        population = new_population
        energies = new_energies
        T = max(T * alpha, T_min)
    print(int(time.time() - start_time), "Seconds")
    print(gen, "Generations")
    return best_turns, best_energy


#----------------------------------#
#-----  Simulated Annealing  ------#
#----------------------------------#


def simulated_annealing(sequence, T, alpha, T_min, iterations = math.inf, time_limit = math.inf):
    start_time = time.time()
    
    turns = ["F"]*(len(sequence) - 2)
    current_energy, best_energy = 0, 0
    best = turns
    
    it = 0
    while it < iterations and ((time.time() - start_time) < time_limit):
        it += 1
    
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
                if new_energy <= best_energy:
                    if new_energy != best_energy:
                        wait = 0
                    best_energy = new_energy
                    best = turns

        T = max(alpha * T, T_min)
    print(int(time.time() - start_time), "Seconds")
    print(it, "Iterations")
    return best, best_energy


#-------------------#
#-----  Main  ------#
#-------------------#


if __name__ == "__main__":
    random.seed(67)
    print("Genetic Algorithm...")
    gen_turns, gen_energy = genetic_algorithm(sequence = "HPHPPHHPHPPHPHHPPHPH", population_size = 3000, T = 3, alpha = 0.995, T_min = 0.1, time_limit = 10)
    print("Simulated Annealing...")
    sim_turns, sim_energy = simulated_annealing(sequence = "HPHPPHHPHPPHPHHPPHPH", T = 3, alpha = 0.995, T_min = 0.1, time_limit = 10)
    print("Genetic Turns:", gen_turns)
    print("Genetic Energy:", gen_energy)
    print("Simulated Annealing Turns:", sim_turns)
    print("Simulated Annealing Energy:", sim_energy)