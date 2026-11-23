import random
import math


#-------------------#
#----- Helper ------#
#-------------------#

def apply_turn(direction, turn):
    rotate = {
        (1, 0): {"L": (0, -1), "F": (1, 0), "R": (0, 1)},
        (0, 1): {"L": (1, 0), "F": (0, 1), "R": (-1, 0)},
        (-1, 0): {"L": (0, 1), "F": (-1, 0), "R": (0, -1)},
        (0, -1): {"L": (-1, 0), "F": (0, -1), "R": (1, 0)}
    }
    return rotate[direction][turn]


def generate_coords(turns):
    coords = [(0, 0), (1, 0)]
    direction = (1, 0)
    for t in turns:
        direction = apply_turn(direction, t)
        position = (coords[-1][0] + direction[0], coords[-1][1] + direction[1])
        coords.append(position)
    return coords


def energy(sequence, coords):
    en = 0
    positions = {coord: i for i, coord in enumerate(coords)}

    for i, (x, y) in enumerate(coords):
        if sequence[i] != "H":
            continue

        neighbours = [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1)
        ]
        
        for neighbour in neighbours:
            if neighbour in positions:
                j = positions[neighbour]

                if j > i and abs(i - j) > 1 and sequence[j] == "H":
                    en -= 1
    return en


def is_valid_walk(coords):
    hash_table = {0: [(0, 0)]}
    prev = (0, 0)
    for row, col in coords[1:]:
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
    return True


#----------------------#
#----- Crossover ------#
#----------------------#


def crossover_probabilities(energies):
    probs = []
    total = sum(energies)
    for energy in energies:
        probs.append(energy / total)
    return probs


def crossover(sequence, mutation_threshold, t1, t2):
    split = random.randrange(len(t1))
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
            return None
    
    e1 = energy(sequence, generate_coords(t1))
    e2 = energy(sequence, generate_coords(t2))
    e_avg = (e1 + e2) / 2
    e3 = energy(sequence, generate_coords(cross))
    if e3 <= e_avg or random.random() < math.exp((e_avg - e3) / mutation_threshold):
        return cross
    return None
        

#---------------------#
#----- Mutation ------#
#---------------------#


def mutate(turns, sequence, mutation_threshold):
    e1 = energy(sequence, generate_coords(turns))
    e2 = None
    self_avoiding = False
    while self_avoiding == False: 
        mutation = turns.copy()
        i = random.randrange(len(turns))
        t = random.choice(["L", "F", "R"])
        if turns[i] == t:
            return turns
        mutation[i] = t
        mutation_coords = generate_coords(mutation)
        if is_valid_walk(mutation_coords):
            self_avoiding = True
            
    e2 = energy(sequence, mutation_coords)
    if e2 <= e1 or random.random() < math.exp((e1 - e2) / mutation_threshold):
        return mutation
    return turns
    

#-------------------#
#-----  Main  ------#
#-------------------#
    

def genetic_algorithm(sequence, population_size, generations, mutation_threshold, alpha, min_mutation_threshold):
    population = [["F"] * (len(sequence) - 2)] * population_size
    best_energy = math.inf
    best_turns = None
    for gen in range(generations):
        new_population = []
        while len(new_population) < population_size:
            p1 = mutate(random.choice(population), sequence, mutation_threshold)
            p2 = mutate(random.choice(population), sequence, mutation_threshold)
            cross = crossover(sequence, mutation_threshold, p1, p2)
            if cross != None:
                new_population.append(cross)
                cross_energy = energy(sequence, generate_coords(cross))
                if cross_energy <= best_energy:
                    best_energy = cross_energy
                    best_turns = cross
        population = new_population
        mutation_threshold = max(mutation_threshold * alpha, min_mutation_threshold)
        if mutation_threshold == min_mutation_threshold:
            print(gen)
            
    print(best_energy)
    print(best_turns)
    return new_population

genetic_algorithm(sequence = "HPHPPHHPHPPHPHHPPHPH", population_size = 500, generations= 100, mutation_threshold = 1.5, alpha = 0.999, min_mutation_threshold = 0.1)