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
    """Calcula fitness de un cromosoma (0 = solución perfecta)"""
    fitness = 0
    
    # 1. Verificar filas (duplicados = errores)
    for row in chromosome:
        fitness += 9 - len(set(row))
    
    # 2. Verificar columnas (duplicados = errores)
    for col in range(9):
        column = [chromosome[row][col] for row in range(9)]
        fitness += 9 - len(set(column))
    
    # 3. Verificar bloques 3x3 (duplicados = errores)
    for block_row in range(0, 9, 3):
        for block_col in range(0, 9, 3):
            block = []
            for r in range(3):
                for c in range(3):
                    block.append(chromosome[block_row + r][block_col + c])
            fitness += 9 - len(set(block))
    
    return fitness

def get_detailed_errors(chromosome):
    """Devuelve errores detallados por celda"""
    errors = [[0 for _ in range(9)] for _ in range(9)]
    
    # Contar errores por fila
    for row in range(9):
        count = {}
        for col in range(9):
            num = chromosome[row][col]
            count[num] = count.get(num, 0) + 1
        
        for col in range(9):
            num = chromosome[row][col]
            if count[num] > 1:
                errors[row][col] += count[num] - 1
    
    # Contar errores por columna
    for col in range(9):
        count = {}
        for row in range(9):
            num = chromosome[row][col]
            count[num] = count.get(num, 0) + 1
        
        for row in range(9):
            num = chromosome[row][col]
            if count[num] > 1:
                errors[row][col] += count[num] - 1
    
    # Contar errores por bloque
    for block_row in range(0, 9, 3):
        for block_col in range(0, 9, 3):
            count = {}
            for r in range(3):
                for c in range(3):
                    num = chromosome[block_row + r][block_col + c]
                    count[num] = count.get(num, 0) + 1
            
            for r in range(3):
                for c in range(3):
                    num = chromosome[block_row + r][block_col + c]
                    if count[num] > 1:
                        errors[block_row + r][block_col + c] += count[num] - 1
    
    return errors

def repair_chromosome(chromosome, initial_board):
    """Repara un cromosoma intercambiando números conflictivos"""
    repaired = deepcopy(chromosome)
    errors = get_detailed_errors(repaired)
    
    # Intentar reparar las celdas con más errores primero
    for _ in range(20):  # Máximo 20 intentos de reparación
        # Encontrar la celda con más errores
        max_errors = -1
        target_row, target_col = -1, -1
        
        for row in range(9):
            for col in range(9):
                if initial_board[row][col] == 0 and errors[row][col] > max_errors:
                    max_errors = errors[row][col]
                    target_row, target_col = row, col
        
        if max_errors <= 0:
            break
        
        # Intentar intercambiar con otra celda en la misma fila
        improved = False
        for swap_col in range(9):
            if (swap_col != target_col and 
                initial_board[target_row][swap_col] == 0 and
                errors[target_row][swap_col] > 0):
                
                # Intercambiar y verificar si mejora
                repaired[target_row][target_col], repaired[target_row][swap_col] = \
                    repaired[target_row][swap_col], repaired[target_row][target_col]
                
                new_fitness = get_fitness(repaired)
                old_fitness = get_fitness(chromosome)
                
                if new_fitness < old_fitness:
                    chromosome = deepcopy(repaired)
                    errors = get_detailed_errors(chromosome)
                    improved = True
                    break
                else:
                    # Revertir si no mejora
                    repaired[target_row][target_col], repaired[target_row][swap_col] = \
                        repaired[target_row][swap_col], repaired[target_row][target_col]
        
        if not improved:
            # Intentar en la misma columna
            for swap_row in range(9):
                if (swap_row != target_row and 
                    initial_board[swap_row][target_col] == 0 and
                    errors[swap_row][target_col] > 0):
                    
                    repaired[target_row][target_col], repaired[swap_row][target_col] = \
                        repaired[swap_row][target_col], repaired[target_row][target_col]
                    
                    new_fitness = get_fitness(repaired)
                    old_fitness = get_fitness(chromosome)
                    
                    if new_fitness < old_fitness:
                        chromosome = deepcopy(repaired)
                        errors = get_detailed_errors(chromosome)
                        improved = True
                        break
                    else:
                        repaired[target_row][target_col], repaired[swap_row][target_col] = \
                            repaired[swap_row][target_col], repaired[target_row][target_col]
        
        if not improved:
            break
    
    return chromosome

def local_search(chromosome, initial_board, iterations=50):
    """Búsqueda local para mejorar un cromosoma"""
    best = deepcopy(chromosome)
    best_fitness = get_fitness(best)
    
    for _ in range(iterations):
        current = deepcopy(best)
        
        # Elegir un bloque 3x3 aleatorio
        block_row = rndm.randint(0, 2) * 3
        block_col = rndm.randint(0, 2) * 3
        
        # Encontrar celdas mutables en este bloque
        mutable_positions = []
        for r in range(3):
            for c in range(3):
                row = block_row + r
                col = block_col + c
                if initial_board[row][col] == 0:
                    mutable_positions.append((row, col))
        
        if len(mutable_positions) >= 2:
            # Intercambiar dos celdas aleatorias en el bloque
            pos1, pos2 = rndm.sample(mutable_positions, 2)
            row1, col1 = pos1
            row2, col2 = pos2
            
            current[row1][col1], current[row2][col2] = \
                current[row2][col2], current[row1][col1]
            
            current_fitness = get_fitness(current)
            
            if current_fitness < best_fitness:
                best = deepcopy(current)
                best_fitness = current_fitness
            elif current_fitness == best_fitness:
                # Aceptar con 20% de probabilidad para diversidad
                if rndm.random() < 0.2:
                    best = deepcopy(current)
                    best_fitness = current_fitness
    
    return best

def tournament_selection(population, tournament_size=5):
    """Selección por torneo (minimización)"""
    selected = []
    for _ in range(len(population)):
        contestants = rndm.sample(population, tournament_size)
        best = min(contestants, key=get_fitness)
        selected.append(deepcopy(best))
    return selected

def crossover_block_swap(ch1, ch2, initial_board, pc=0.85):
    """Crossover intercambiando dos bloques 3x3 aleatorios"""
    if rndm.random() > pc:
        return deepcopy(ch1), deepcopy(ch2)
    
    child1 = deepcopy(ch1)
    child2 = deepcopy(ch2)
    
    # Elegir dos bloques diferentes para intercambiar
    block1 = rndm.randint(0, 2) * 3
    block2 = rndm.randint(0, 2) * 3
    while block2 == block1:
        block2 = rndm.randint(0, 2) * 3
    
    # Intercambiar filas de bloques completos (3 filas cada bloque)
    for offset in range(3):
        row1 = block1 + offset
        row2 = block2 + offset
        
        for col in range(9):
            if initial_board[row1][col] == 0 and initial_board[row2][col] == 0:
                child1[row1][col], child1[row2][col] = child1[row2][col], child1[row1][col]
                child2[row1][col], child2[row2][col] = child2[row2][col], child2[row1][col]
    
    return child1, child2

def adaptive_mutation(chromosome, initial_board, pm_base=0.15, gen=0, max_gen=1000):
    """Mutación adaptativa que disminuye con el tiempo"""
    mutated = deepcopy(chromosome)
    
    # Enfriamiento: reducir mutación gradualmente
    cooling_factor = 1.0 - (gen / max_gen) * 0.5
    pm = pm_base * cooling_factor
    
    if rndm.random() > pm:
        return mutated
    
    # Elegir tipo de mutación basado en progreso
    mutation_type = rndm.random()
    
    if mutation_type < 0.4:  # Intercambio en fila
        row = rndm.randint(0, 8)
        mutable_positions = []
        for col in range(9):
            if initial_board[row][col] == 0:
                mutable_positions.append(col)
        
        if len(mutable_positions) >= 2:
            col1, col2 = rndm.sample(mutable_positions, 2)
            mutated[row][col1], mutated[row][col2] = mutated[row][col2], mutated[row][col1]
    
    elif mutation_type < 0.7:  # Intercambio en columna
        col = rndm.randint(0, 8)
        mutable_positions = []
        for row in range(9):
            if initial_board[row][col] == 0:
                mutable_positions.append(row)
        
        if len(mutable_positions) >= 2:
            row1, row2 = rndm.sample(mutable_positions, 2)
            mutated[row1][col], mutated[row2][col] = mutated[row2][col], mutated[row1][col]
    
    else:  # Intercambio en bloque 3x3
        block_row = rndm.randint(0, 2) * 3
        block_col = rndm.randint(0, 2) * 3
        
        mutable_positions = []
        for r in range(3):
            for c in range(3):
                row = block_row + r
                col = block_col + c
                if initial_board[row][col] == 0:
                    mutable_positions.append((row, col))
        
        if len(mutable_positions) >= 2:
            pos1, pos2 = rndm.sample(mutable_positions, 2)
            row1, col1 = pos1
            row2, col2 = pos2
            mutated[row1][col1], mutated[row2][col2] = mutated[row2][col2], mutated[row1][col1]
    
    return mutated

def elitism_with_memory(old_pop, new_pop, elite_percentage=0.15, memory_size=10):
    """Elitismo que mantiene memoria de las mejores soluciones"""
    combined = old_pop + new_pop
    combined.sort(key=get_fitness)
    
    # Mantener las mejores soluciones únicas
    unique_best = []
    seen = set()
    
    for ch in combined:
        ch_tuple = tuple(tuple(row) for row in ch)
        if ch_tuple not in seen:
            seen.add(ch_tuple)
            unique_best.append(ch)
            if len(unique_best) >= len(old_pop):
                break
    
    # Si no hay suficientes únicos, completar con los mejores repetidos
    if len(unique_best) < len(old_pop):
        unique_best.extend(combined[:len(old_pop) - len(unique_best)])
    
    return unique_best[:len(old_pop)]

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
                            pm_base=0.15, pc=0.85, elite_percentage=0.15):
    initial_board = read_puzzle(initial_file)
    
    # Crear población inicial
    population = make_population(pop_size, initial_board)
    best_fitness_history = []
    stagnation_counter = 0
    last_best_fitness = float('inf')
    
    # Aplicar reparación inicial a toda la población
    print("Aplicando reparación inicial a la población")
    for i in range(len(population)):
        population[i] = repair_chromosome(population[i], initial_board)
    
    best_solution = None
    best_fitness_overall = float('inf')
    
    for gen in range(generations):
        # Calcular fitness
        fitness_values = [get_fitness(ch) for ch in population]
        best_fitness = min(fitness_values)
        best_idx = fitness_values.index(best_fitness)
        
        # Guardar la mejor solución global
        if best_fitness < best_fitness_overall:
            best_fitness_overall = best_fitness
            best_solution = deepcopy(population[best_idx])
        
        best_fitness_history.append(best_fitness)
        
        # Condición de terminación
        if best_fitness == 0:
            print(f"Solución perfecta encontrada en generación {gen}")
            return population[best_idx]
        
        # Detectar estancamiento
        if best_fitness >= last_best_fitness:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
        
        last_best_fitness = best_fitness
        
        # Aplicar búsqueda local si hay estancamiento
        if stagnation_counter > 30 and gen > 100:
            print(f"Generación {gen}: Estancamiento detectado. Aplicando búsqueda local")
            # Aplicar búsqueda local a los mejores individuos
            for i in range(min(10, len(population))):
                population[i] = local_search(population[i], initial_board, 100)
            stagnation_counter = 0
        
        # Aplicar reparación periódica
        if gen % 50 == 0 and gen > 0:
            for i in range(min(50, len(population))):
                population[i] = repair_chromosome(population[i], initial_board)
        
        # Selección
        selected = tournament_selection(population, 5)
        
        # Cruzamiento y mutación
        offspring = []
        for i in range(0, len(selected), 2):
            if i + 1 < len(selected):
                child1, child2 = crossover_block_swap(
                    selected[i], selected[i+1], initial_board, pc
                )
                offspring.append(adaptive_mutation(child1, initial_board, pm_base, gen, generations))
                offspring.append(adaptive_mutation(child2, initial_board, pm_base, gen, generations))
        
        # Aplicar elitismo con memoria
        population = elitism_with_memory(population, offspring, elite_percentage)
        
        # Reporte periódico
        if gen % 50 == 0:
            avg_fitness = sum(fitness_values) / len(fitness_values)
            best_chromosome = population[best_idx]
            
            # Calcular errores por tipo
            row_errors = sum([9 - len(set(row)) for row in best_chromosome])
            col_errors = sum([9 - len(set([best_chromosome[r][c] for r in range(9)])) for c in range(9)])
            block_errors = 0
            for br in range(0, 9, 3):
                for bc in range(0, 9, 3):
                    block = []
                    for r in range(3):
                        for c in range(3):
                            block.append(best_chromosome[br + r][bc + c])
                    block_errors += 9 - len(set(block))
            
            print(f"Gen {gen:4d}: Mejor={best_fitness:3d}, Prom={avg_fitness:6.1f}, "
                  f"Errores[F:{row_errors:2d} C:{col_errors:2d} B:{block_errors:2d}]")
    
    # Retornar la mejor solución encontrada
    print(f"\nMejor fitness alcanzado: {best_fitness_overall}")
    
    # Intentar una última reparación intensiva
    if best_fitness_overall > 0:
        print("Aplicando reparación final intensiva")
        for _ in range(100):
            repaired = repair_chromosome(best_solution, initial_board)
            new_fitness = get_fitness(repaired)
            if new_fitness < best_fitness_overall:
                best_solution = repaired
                best_fitness_overall = new_fitness
                if best_fitness_overall == 0:
                    print("Reparación final encontró solución perfecta")
                    break
    
    return best_solution

# Ejecución
if __name__ == "__main__":
    tic = time.time()
    
    solution = genetic_algorithm_sudoku(
        initial_file="./Test.txt",
        pop_size=800,
        generations=1500,  # Más generaciones
        pm_base=0.18,
        pc=0.88,
        elite_percentage=0.15
    )
    
    toc = time.time()
    
    final_fitness = get_fitness(solution)
    print(f"\nTiempo total: {toc - tic:.2f} segundos")
    print(f"Fitness final: {final_fitness}")
    
    if final_fitness == 0:
        print("\nSolucion del Sudoku encontrada:")
    else:
        print("\nSolución parcial encontrada (mejor intento):")
        # Mostrar distribución de errores
        print("\nDistribución de errores:")
        errors = get_detailed_errors(solution)
        total_errors = sum(sum(row) for row in errors)
        print(f"Errores totales en celdas: {total_errors}")
        
        # Mostrar filas con errores
        print("\nErrores por fila:")
        for i, row in enumerate(solution):
            row_errors = 9 - len(set(row))
            if row_errors > 0:
                print(f"Fila {i+1}: {row_errors} errores - {row}")
    
    print("\nMejor solución encontrada:")
    print_chromosome(solution)