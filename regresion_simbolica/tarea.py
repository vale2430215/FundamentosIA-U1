# CELDA 1 Importación de librerías esenciales
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets, model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from deap import base, creator, tools, algorithms

# Configuración de semilla aleatoria para reproducibilidad
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

#CELDA 2
# Cargar el dataset Wine desde scikit-learn
wine = datasets.load_wine()
X, y = wine.data, wine.target

print(f"Dimensiones de X (características): {X.shape}")
print(f"Dimensiones de y (etiquetas): {y.shape}")
print(f"Nombres de las características: {wine.feature_names}")

# Configurar validación cruzada (k-fold = 5)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

#CELDA 3
default_clf = AdaBoostClassifier(random_state=RANDOM_SEED)
scores_default = model_selection.cross_val_score(default_clf, X, y, cv=kfold, scoring='accuracy')
acc_default = scores_default.mean()

print(f"Precisión media con hiperparámetros por defecto: {acc_default:.4f} ({acc_default*100:.2f}%)")

#CELDA 4
# Rejilla actualizada sin el parámetro 'algorithm'
grid_params = {
    'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'learning_rate': np.logspace(-2, 0, num=10, base=10)
}

start_time = time.time()
grid_search = GridSearchCV(
    estimator=AdaBoostClassifier(random_state=RANDOM_SEED),
    param_grid=grid_params,
    cv=kfold,
    scoring='accuracy',
    n_jobs=-1  # Utiliza todos los núcleos disponibles del procesador
)
grid_search.fit(X, y)
elapsed_grid = time.time() - start_time

acc_grid = grid_search.best_score_
print(f"GridSearchCV - Mejores Parámetros: {grid_search.best_params_}")
print(f"GridSearchCV - Mejor Precisión: {acc_grid:.4f} ({acc_grid*100:.2f}%)")
print(f"GridSearchCV - Tiempo Transcurrido: {elapsed_grid:.2f} segundos")

#CELDA 5
# Definición del espacio de búsqueda y conversiones
BOUNDS_LOW = [1.0, 0.01, 0.0]
BOUNDS_HIGH = [100.0, 1.00, 1.0]
NUM_PARAMS = len(BOUNDS_LOW)
CROWDING_FACTOR = 20.0

def convert_params(params):
    """Convierte el genotipo de flotantes a hiperparámetros reales del clasificador."""
    n_estimators = int(round(params[0]))
    learning_rate = params[1]
    algo_idx = int(round(params[2]))
    algorithm = ['SAMME', 'SAMME.R'][algo_idx]
    return n_estimators, learning_rate, algorithm

def evaluate_accuracy(individual):
    """Función de fitness: evalúa la precisión con k-fold cross validation."""
    n_est, lr, algo = convert_params(individual)
    clf = AdaBoostClassifier(
        n_estimators=n_est,
        learning_rate=lr,
        algorithm=algo,
        random_state=RANDOM_SEED
    )
    scores = model_selection.cross_val_score(clf, X, y, cv=kfold, scoring='accuracy')
    return (scores.mean(),)

# Configuración del Framework DEAP
if hasattr(creator, "FitnessMax"):
    del creator.FitnessMax
if hasattr(creator, "Individual"):
    del creator.Individual

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Generadores para cada hiperparámetro según sus límites
for i in range(NUM_PARAMS):
    toolbox.register(f"param_{i}", random.uniform, BOUNDS_LOW[i], BOUNDS_HIGH[i])

param_generators = tuple(getattr(toolbox, f"param_{i}") for i in range(NUM_PARAMS))
toolbox.register("individualCreator", tools.initCycle, creator.Individual, param_generators, n=1)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)
toolbox.register("evaluate", evaluate_accuracy)

# Operadores Genéticos
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUNDS_LOW, up=BOUNDS_HIGH, eta=CROWDING_FACTOR)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUNDS_LOW, up=BOUNDS_HIGH, eta=CROWDING_FACTOR, indpb=1.0/NUM_PARAMS)

# Implementación de Elitismo
def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=True):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    for gen in range(1, ngen + 1):
        offspring = toolbox.select(population, len(population))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for i in range(1, len(offspring), 2):
            if random.random() < cxpb:
                toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values
                del offspring[i].fitness.values

        for i in range(len(offspring)):
            if random.random() < mutpb:
                toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(offspring)
            offspring[:len(halloffame)] = [toolbox.clone(ind) for ind in halloffame]

        population[:] = offspring
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

#CELDA 6
# Parámetros del Algoritmo Genético
POPULATION_SIZE = 20
P_CROSSOVER = 0.9
P_MUTATION = 0.1
MAX_GENERATIONS = 10
HALL_OF_FAME_SIZE = 3

population = toolbox.populationCreator(n=POPULATION_SIZE)
hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("avg", np.mean)

start_time = time.time()
population, logbook = eaSimpleWithElitism(
    population,
    toolbox,
    cxpb=P_CROSSOVER,
    mutpb=P_MUTATION,
    ngen=MAX_GENERATIONS,
    stats=stats,
    halloffame=hof,
    verbose=True
)
elapsed_ga = time.time() - start_time

best_ind = hof.items[0]
best_n_est, best_lr = int(best_ind[0]), float(best_ind[1])
acc_ga = best_ind.fitness.values[0]

print("\n--- RESULTADOS DEL ALGORITMO GENÉTICO ---")
print(f"Mejores hiperparámetros: n_estimators={best_n_est}, learning_rate={best_lr:.4f}")
print(f"Mejor precisión obtenida: {acc_ga:.4f} ({acc_ga*100:.2f}%)")
print(f"Tiempo transcurrido: {elapsed_ga:.2f} segundos")

#CELDA 7
methods = ['Por Defecto', 'GridSearchCV', 'Algoritmo Genético (DEAP)']
accuracies = [acc_default * 100, acc_grid * 100, acc_ga * 100]
colors = ['#d95f02', '#7570b3', '#1b9e77']

plt.figure(figsize=(9, 5))
bars = plt.bar(methods, accuracies, color=colors, width=0.5)
plt.ylabel('Precisión (%)')
plt.title('Comparación de Precisión en UCI Wine Dataset (AdaBoost)')
plt.ylim(50, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()