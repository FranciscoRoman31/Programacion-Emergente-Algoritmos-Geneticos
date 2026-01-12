# Documentación del Proyecto: Algoritmos Genéticos

Este proyecto implementa dos algoritmos genéticos para resolver diferentes problemas:
1. **Sodoku**: Resolución de puzzles de Sudoku
2. **Horarios**: Generación de horarios escolares/universitarios

# Archivos del Proyecto

### 1. `soduku.py` - Solucionador de Sudoku con Algoritmo Genético

#### Descripción
Implementa un algoritmo genético para resolver puzzles de Sudoku. El algoritmo evoluciona una población de soluciones potenciales hasta encontrar una solución válida.

#### Funciones Principales

##### `make_gene(initial_row)`
Crea una fila (gen) que respeta los números fijos iniciales del Sudoku.
- **Parámetros**:
  - `initial_row`: Lista de 9 elementos (0 para vacíos, 1-9 para fijos)
- **Retorna**: Lista de 9 números que forman una fila válida

##### `make_chromosome(initial_board)`
Crea un cromosoma completo (tablero de Sudoku) respetando todos los números fijos.
- **Parámetros**:
  - `initial_board`: Matriz 9x9 con el puzzle inicial
- **Retorna**: Matriz 9x9 completa

##### `make_population(count, initial_board)`
Genera una población inicial de cromosomas.
- **Parámetros**:
  - `count`: Número de individuos en la población
  - `initial_board`: Puzzle inicial
- **Retorna**: Lista de cromosomas

##### `get_fitness(chromosome)`
Calcula el fitness de un cromosoma. Fitness máximo = 243 (27 sets × 9 elementos).
- **Puntos por**: 
  - Filas únicas: 81 puntos
  - Columnas únicas: 81 puntos
  - Bloques 3x3 únicos: 81 puntos
- **Retorna**: Valor numérico de fitness

##### `tournament_selection(population, tournament_size=3)`
Selección por torneo: selecciona los mejores de grupos aleatorios.
- **Parámetros**:
  - `population`: Población actual
  - `tournament_size`: Tamaño del torneo
- **Retorna**: Población seleccionada

##### `crossover(ch1, ch2, initial_board, pc=0.8)`
Operador de cruce de un punto entre bloques 3x3.
- **Características**: Respeta números fijos del puzzle original

##### `mutation(chromosome, initial_board, pm=0.1)`
Mutación por intercambio en la misma fila.
- **Característica**: Solo intercambia posiciones que eran vacías originalmente

##### `elitism(old_pop, new_pop, elite_percentage=0.1)`
Preserva los mejores individuos entre generaciones.

##### `genetic_algorithm_sudoku()`
Función principal que ejecuta el algoritmo genético completo.
- **Parámetros configurables**:
  - `pop_size`: Tamaño de la población (default: 500)
  - `generations`: Número máximo de generaciones (default: 500)
  - `pm`: Probabilidad de mutación (default: 0.15)
  - `pc`: Probabilidad de cruce (default: 0.85)
  - `elite_percentage`: Porcentaje de élite (default: 0.1)

#### Formato del Archivo de Entrada
El archivo debe contener 9 líneas, cada una con 9 números separados por espacios:
- `0` representa celdas vacías
- `1-9` representa números fijos

#### Ejemplo de Uso
```python
solution = genetic_algorithm_sudoku(
    initial_file="./Test.txt",
    pop_size=500,
    generations=500
)
```

---

### 2. `horarios.py` - Generador de Horarios con Algoritmo Genético

#### Descripción
Implementa un algoritmo genético para generar horarios escolares/universitarios evitando conflictos entre profesores y aulas.

#### Estructura de Datos
- **Clases**: Historia, Química, Física, Matemática, Biología, Inglés, Educación Física, Informática
- **Profesores**: Angel, Maria, Luis Miguel, Chayanne
- **Aulas**: A1, A2, A3
- **Horas**: Lunes 8pm, Lunes 10pm, Martes 8pm, Martes 10pm, Miércoles 8pm, Miércoles 10pm, Jueves 8pm, Viernes 10pm

#### Funciones Principales

##### `crear_poblacion(size)`
Crea una población inicial de horarios aleatorios.
- **Parámetros**: `size` - Tamaño de la población
- **Retorna**: Lista de horarios (cada horario es lista de 8 clases)

##### `fitness(horario)`
Calcula el fitness de un horario.
- **Reglas**:
  - Penaliza si un profesor tiene dos clases a la misma hora
  - Penaliza si un aula es usada por dos clases a la misma hora
- **Fitness ideal**: 0 (sin conflictos)

##### `seleccion(poblacion)`
Selección por torneo (tamaño 3).
- **Retorna**: El mejor individuo de 3 aleatorios

##### `crossover(p1, p2)`
Cruce de un punto entre dos horarios.
- **Parámetros**: Dos horarios padres
- **Retorna**: Un horario hijo

##### `mutacion(horario, prob=0.1)`
Aplica mutación aleatoria a un horario.
- **Probabilidad**: 10% por defecto

#### Ejemplo de Uso
```python
poblacion = crear_poblacion(3)
for gen in range(50):
    # Evolución de la población
    mejor = max(poblacion, key=fitness)
    if fitness(mejor) == 0:
        break
```

---

## Requisitos del Sistema

### Dependencias
```bash
pip install matplotlib prettyprinter
```

### Versión de Python
Python 3.6 o superior

## Ejecución

### Para Sudoku:
```bash
python soduku.py
```
**Nota**: Requiere archivo `Test.txt` con el puzzle inicial

### Para Horarios:
```bash
python horarios.py
```

---

## Parámetros Ajustables

### Sudoku (`soduku.py`):
- `pop_size`: Tamaño de población (100-1000 recomendado)
- `generations`: Generaciones máximas (500-5000)
- `pm`: Probabilidad de mutación (0.05-0.2)
- `pc`: Probabilidad de cruce (0.7-0.95)

### Horarios (`horarios.py`):
- Tamaño de población: Modificar en `crear_poblacion()`
- Probabilidad de mutación: Modificar en `mutacion(prob=0.1)`
- Generaciones máximas: Modificar en el bucle `for gen in range(50)`

---

## Resultados Esperados

### Sudoku:
- Imprime la solución encontrada
- Muestra fitness final (243 = solución perfecta)
- Tiempo de ejecución

### Horarios:
- Imprime el mejor horario de cada generación
- Muestra fitness (0 = sin conflictos)
- Formato pretty-printed para mejor legibilidad

---

## Limitaciones y Consideraciones

### Sudoku:
- Puede no encontrar solución en el número de generaciones dado
- El tiempo de ejecución depende de la dificultad del puzzle
- Probabilístico: diferentes ejecuciones pueden dar resultados diferentes

### Horarios:
- Número fijo de clases (8)
- No considera preferencias de profesores
- Solo evita conflictos básicos de tiempo

---

## Mejoras Futuras

### Para Sudoku:
1. Implementar operadores de cruce más inteligentes
2. Añadir técnicas de reparación de soluciones inválidas
3. Implementar paralelización para poblaciones grandes

### Para Horarios:
1. Añadir más restricciones (preferencias de profesores, niveles de clase)
2. Implementar interfaz gráfica
3. Permitir configuración dinámica de clases/profesores/aulas
4. Añadir visualización de horarios

---
Proyecto de algoritmos genéticos para resolución de problemas de optimización.

