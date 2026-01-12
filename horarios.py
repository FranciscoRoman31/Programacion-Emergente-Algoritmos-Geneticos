import random
import matplotlib as plt
from prettyprinter import pprint

clases = ['Historia', 'Quimica', 'Fisica', 'Matematica', 'Biologia', 'Ingles', 'Educacion Fisica', 'Informatica']
profesores = ['Angel', 'Maria', 'Luis Miguel', 'Chayanne']
aulas = ['A1', 'A2', 'A3']
horas = ['Lunes 8pm', 'Lunes 10pm', 'Martes 8pm', 'Martes 10pm', 'Miercoles 8pm', 'Miercoles 10pm', 'Jueves 8pm', 'Viernes 10pm']

def crear_poblacion(size):
    return [[(random.choice(clases),
              random.choice(profesores),
              random.choice(aulas),
              random.choice(horas)) for _ in clases] for _ in range(size)]

def fitness(horario):
    score = 0
    used_prof = {}
    used_aulas = {}
    for clase, prof, aula, hora in horario:
        if (prof, hora) in used_prof:
            score -= 1
        else:
            used_prof[(prof, hora)] = True

        if (aula, hora) in used_aulas:
            score -= 1
        else:
            used_aulas[(aula, hora)] = True
    return score

def seleccion(poblacion):
    return max(random.sample(poblacion, 3), key = fitness)

def crossover(p1, p2):
    punto = random.randint(1, len(p1) - 1)
    return p1[punto:] + p2[:punto]

def mutacion(horario, prob = 0.1):
    if random.random() < prob:
        i = random.randint(0, len(horario) - 1)
        clase, prof, aula, hora = horario[i]
        horario[i] = (clase,
                      random.choice(profesores),
                      random.choice(aulas),
                      random.choice(horas))
    return horario

def crear_graficas(horario):
    pass

poblacion = crear_poblacion(3)
print('Poblacion inicial')
for valor in poblacion:
    pprint(valor)
print("\n")
for gen in range(50):
    nueva_poblacion = []
    for _ in range(len(poblacion)):
        p1, p2 = seleccion(poblacion), seleccion(poblacion)
        hijo = crossover(p1, p2)
        hijo = mutacion(hijo)
        nueva_poblacion.append(hijo)
    poblacion = nueva_poblacion
    mejor = max(poblacion, key = fitness)
    #print(f"Gen: {gen}: mejor fitness = {fitness(mejor)}, Horario: {mejor}")
    data = {
        "gen": gen,
        "Mejor fitness": fitness(mejor),
        "Horario": mejor
    }
    pprint(data)
    if fitness(mejor) == 0:
        break