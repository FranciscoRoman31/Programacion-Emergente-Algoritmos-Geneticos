import random
import matplotlib.pyplot as plt
from prettyprinter import pprint

clases = ['Historia', 'Quimica', 'Fisica', 'Matematica', 'Biologia', 'Ingles', 'Educacion Fisica', 'Informatica']
profesores = ['Angel', 'Maria', 'Luis Miguel', 'Chayanne']
aulas = ['A1', 'A2']
#horas = ['Lunes 8pm', 'Lunes 10pm', 'Martes 8pm', 'Martes 10pm', 'Miercoles 8pm', 'Miercoles 10pm', 'Jueves 8pm', 'Viernes 10pm']
fechas = ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes']
horas = ["8pm", "10pm"]
registro_evolutivo = []

def crear_poblacion(size):
    return [[(random.choice(clases),
              random.choice(profesores),
              random.choice(aulas),
              random.choice(fechas),
              random.choice(horas)) for _ in clases] for _ in range(size)]

def fitness(horario):
    score = 0
    used_prof = {}
    used_aulas = {}
    for clase, prof, aula, fecha, hora in horario:
        if (prof, fecha, hora) in used_prof:
            score -= 1
        else:
            used_prof[(prof, fecha, hora)] = True

        if (aula, fecha, hora) in used_aulas:
            score -= 1
        else:
            used_aulas[(aula, fecha, hora)] = True
    return score

def seleccion(poblacion):
    return max(random.sample(poblacion, 3), key = fitness)

def crossover(p1, p2):
    punto = random.randint(1, len(p1) - 1)
    return p1[punto:] + p2[:punto]

def mutacion(horario, prob = 0.1):
    if random.random() < prob:
        i = random.randint(0, len(horario) - 1)
        clase, prof, aula, fecha, hora = horario[i]
        horario[i] = (clase,
                      random.choice(profesores),
                      random.choice(aulas),
                      random.choice(fechas),
                      random.choice(horas))
    return horario

def crear_graficas_proceso_evolutivo(gen_data):
    generaciones = [data["gen"] for data in gen_data]
    fitness_values = [data["Mejor fitness"] for data in gen_data]

    plt.figure(figsize=(10, 5))
    plt.plot(generaciones, fitness_values, marker='o')
    plt.title('Evolución del Fitness del Mejor Horario')
    plt.xlabel('Generación')
    plt.ylabel('Fitness del Mejor Horario')
    plt.grid()
    plt.show()

def mostrar_horario_completo(horario):
    horas_dict = {}
    for clase, prof, aula, fecha, hora in horario:
        key = f"{fecha} {hora}"
        if key not in horas_dict:
            horas_dict[key] = []
        horas_dict[key].append(f"{clase} ({prof}, {aula})")

    fig, ax = plt.subplots(figsize=(11, 5))
    y_pos = range(len(horas))

    for i, hora in enumerate(horas):
        for j, fecha in enumerate(fechas):
            key = f"{fecha} {hora}"
            if key in horas_dict:
                clases_info = "\n".join(horas_dict[key])
                ax.broken_barh([(j, 0.8)], (i - 0.4, 0.8), facecolors='lightgrey', edgecolors='black', linewidth=1, zorder=1)
                ax.text(j + 0.4, i, clases_info, ha='center', va='center', fontsize=8, color='black')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(horas)
    ax.invert_yaxis()

    
    ax.set_xticks([i + 0.4 for i in range(len(fechas))])
    ax.set_xticklabels(fechas)
    ax.set_xlim(-0.1, len(fechas) + 0.2)

    ax.set_title('Horario Docente')

    plt.tight_layout()
    plt.show()

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
    registro_evolutivo.append(data)

    pprint(data)
    if fitness(mejor) == 0:
        crear_graficas_proceso_evolutivo(registro_evolutivo)
        mostrar_horario_completo(mejor)
        break