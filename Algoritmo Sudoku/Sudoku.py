import random as rndm
import time
from copy import deepcopy

def make_gene(initial_row):
    """
    Crea una fila (gen) que respete los números fijos iniciales.
    initial_row: lista de 9 elementos, 0 para vacíos, 1-9 para fijos
    """
    gene = [0] * 9
    used_numbers = set()
    
    # Primero colocar números fijos
    for i in range(9):
        if initial_row[i] != 0:
            gene[i] = initial_row[i]
            used_numbers.add(initial_row[i])
    
    # Llenar vacíos con números no usados
    available = [num for num in range(1, 10) if num not in used_numbers]
    rndm.shuffle(available)
    
    idx = 0
    for i in range(9):
        if gene[i] == 0:
            gene[i] = available[idx]
            idx += 1
    
    return gene

def make_chromosome(initial_board):
    """Crea un cromosoma completo respetando números fijos"""
    chromosome = []
    for row in initial_board:
        chromosome.append(make_gene(row))
    return chromosome

def make_population(count, initial_board):
    """Crea población inicial"""
    population = []
    for _ in range(count):
        population.append(make_chromosome(initial_board))
    return population

def get_fitness(chromosome):
    """Calcula fitness de un cromosoma."""
    fitness = 0
    
    # 1. Verificar filas (81 puntos)
    for row in chromosome:
        fitness += len(set(row))
    
    # 2. Verificar columnas (81 puntos)
    for col in range(9):
        column = [chromosome[row][col] for row in range(9)]
        fitness += len(set(column))
    
    # 3. Verificar bloques 3x3 (81 puntos)
    for block_row in range(0, 9, 3):
        for block_col in range(0, 9, 3):
            block = []
            for r in range(3):
                for c in range(3):
                    block.append(chromosome[block_row + r][block_col + c])
            fitness += len(set(block))
    
    return fitness

def tournament_selection(population, tournament_size=3):
    """Selección por torneo más eficiente"""
    selected = []
    for _ in range(len(population)):
        contestants = rndm.sample(population, tournament_size)
        best = max(contestants, key=get_fitness)
        selected.append(deepcopy(best))
    return selected

def crossover(ch1, ch2, initial_board, pc=0.8):
    """Crossover de un punto entre bloques 3x3"""
    if rndm.random() > pc:
        return deepcopy(ch1), deepcopy(ch2)
    
    child1 = deepcopy(ch1)
    child2 = deepcopy(ch2)
    
    # Cruzar por bloques de filas
    crossover_point = rndm.randint(1, 8)
    
    for i in range(9):
        if i >= crossover_point:
            # Solo intercambiar si no viola números fijos
            for j in range(9):
                if initial_board[i][j] == 0:  # Solo celdas vacías
                    child1[i][j], child2[i][j] = child2[i][j], child1[i][j]
    
    return child1, child2

def mutation(chromosome, initial_board, pm=0.1):
    """Mutación por intercambio en la misma fila"""
    mutated = deepcopy(chromosome)
    
    if rndm.random() > pm:
        return mutated
    
    # Seleccionar una fila aleatoria
    row = rndm.randint(0, 8)
    
    # Encontrar dos posiciones en la misma fila que sean mutables
    mutable_positions = []
    for col in range(9):
        if initial_board[row][col] == 0:
            mutable_positions.append(col)
    
    if len(mutable_positions) >= 2:
        col1, col2 = rndm.sample(mutable_positions, 2)
        mutated[row][col1], mutated[row][col2] = mutated[row][col2], mutated[row][col1]
    
    return mutated

def elitism(old_pop, new_pop, elite_percentage=0.1):
    """Preserva los mejores individuos"""
    combined = old_pop + new_pop
    combined.sort(key=get_fitness, reverse=True)
    return combined[:len(old_pop)]

def read_puzzle(address):
    """Lee el puzzle desde archivo"""
    puzzle = []
    with open(address, 'r') as f:
        for row in f:
            puzzle.append([int(num) for num in row.split()])
    return puzzle

def print_chromosome(ch):
    """Imprime el cromosoma formateado"""
    for i in range(9):
        for j in range(9):
            print(ch[i][j], end=" ")
        print()

def genetic_algorithm_sudoku(initial_file, pop_size=1000, generations=1000, 
                            pm=0.1, pc=0.8, elite_percentage=0.1):
    initial_board = read_puzzle(initial_file)
    
    # Crear población inicial
    population = make_population(pop_size, initial_board)
    best_fitness_history = []
    
    for gen in range(generations):
        # Calcular fitness
        fitness_values = [get_fitness(ch) for ch in population]
        best_fitness = max(fitness_values)
        best_fitness_history.append(best_fitness)
        
        # Condición de terminación
        if best_fitness == 243:
            print(f"Solución encontrada en generación {gen}")
            best_idx = fitness_values.index(best_fitness)
            return population[best_idx]
        
        # Selección
        selected = tournament_selection(population)
        
        # Cruzamiento y mutación
        offspring = []
        for i in range(0, len(selected), 2):
            if i + 1 < len(selected):
                child1, child2 = crossover(
                    selected[i], selected[i+1], initial_board, pc
                )
                offspring.append(mutation(child1, initial_board, pm))
                offspring.append(mutation(child2, initial_board, pm))
        
        # Aplicar elitismo
        population = elitism(population, offspring, elite_percentage)
        
        # Reporte periódico
        if gen % 100 == 0:
            avg_fitness = sum(fitness_values) / len(fitness_values)
            print(f"Gen {gen}: Mejor={best_fitness}, Promedio={avg_fitness:.1f}")
    
    # Retornar el mejor de la última generación
    fitness_values = [get_fitness(ch) for ch in population]
    best_idx = fitness_values.index(max(fitness_values))
    return population[best_idx]

# Ejecución
if __name__ == "__main__":
    tic = time.time()
    solution = genetic_algorithm_sudoku(
        initial_file="./Test.txt",
        pop_size=500,
        generations=500,
        pm=0.15,
        pc=0.85,
        elite_percentage=0.1
    )
    toc = time.time()
    
    print(f"\nTiempo total: {toc - tic:.2f} segundos")
    print(f"Fitness final: {get_fitness(solution)}")
    print("\nMejor solución encontrada:")
    print_chromosome(solution)