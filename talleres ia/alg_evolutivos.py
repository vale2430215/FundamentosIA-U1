import random
from collections import deque

import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)

maze = [
    [1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,1,0,0,0,0,1],
    [1,0,1,0,1,0,1,1,0,1],
    [1,0,1,0,0,0,0,1,0,1],
    [1,0,1,1,1,1,0,1,0,1],
    [1,0,0,0,0,1,0,0,0,1],
    [1,1,1,1,0,1,1,1,0,1],
    [1,0,0,0,0,0,0,1,0,1],
    [1,0,1,1,1,1,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1],
]

START = (1, 1)
GOAL = (8, 8)

ROWS = len(maze)
COLS = len(maze[0])

MOVES = {
    "U": (-1, 0),  # arriba
    "D": (1, 0),   # abajo
    "L": (0, -1),  # izquierda
    "R": (0, 1),   # derecha
}

def plot_maze(path=None, title="Laberinto"):
    plt.figure(figsize=(6, 6))
    plt.imshow(maze, cmap="gray_r")

    # Inicio y meta
    plt.scatter(START[1], START[0], s=120, marker="o", label="Inicio")
    plt.scatter(GOAL[1], GOAL[0], s=160, marker="*", label="Meta")

    # Camino recorrido
    if path:
        ys = [position[0] for position in path]
        xs = [position[1] for position in path]
        plt.plot(xs, ys, linewidth=2, label="Recorrido")

    plt.xticks(range(COLS))
    plt.yticks(range(ROWS))
    plt.grid(alpha=0.2)
    plt.title(title)
    plt.legend()
    plt.show()


plot_maze()

CHROMOSOME_LENGTH = 20


def create_individual():
    return [
        random.choice(list(MOVES.keys()))
        for _ in range(CHROMOSOME_LENGTH)
    ]


example_individual = create_individual()
example_individual

def simulate_individual(individual):
    row, col = START
    path = [START]
    invalid_moves = 0

    for step, move in enumerate(individual, start=1):
        dr, dc = MOVES[move]
        new_row = row + dr
        new_col = col + dc

        if (
            0 <= new_row < ROWS
            and 0 <= new_col < COLS
            and maze[new_row][new_col] == 0
        ):
            row, col = new_row, new_col
            path.append((row, col))
        else:
            invalid_moves += 1

        if (row, col) == GOAL:
            return {
                "reached_goal": True,
                "position": (row, col),
                "path": path,
                "invalid_moves": invalid_moves,
                "steps_used": step,
            }

    return {
        "reached_goal": False,
        "position": (row, col),
        "path": path,
        "invalid_moves": invalid_moves,
        "steps_used": CHROMOSOME_LENGTH,
    }


simulate_individual(example_individual)

def calculate_bfs_distances():
    distances = [
        [None for _ in range(COLS)]
        for _ in range(ROWS)
    ]

    queue = deque([GOAL])
    distances[GOAL[0]][GOAL[1]] = 0

    while queue:
        row, col = queue.popleft()

        for dr, dc in MOVES.values():
            new_row = row + dr
            new_col = col + dc

            if (
                0 <= new_row < ROWS
                and 0 <= new_col < COLS
                and maze[new_row][new_col] == 0
                and distances[new_row][new_col] is None
            ):
                distances[new_row][new_col] = distances[row][col] + 1
                queue.append((new_row, new_col))

    return distances


BFS_DISTANCES = calculate_bfs_distances()
START_DISTANCE = BFS_DISTANCES[START[0]][START[1]]

print("Distancia mínima desde el inicio hasta la meta:", START_DISTANCE)

INVALID_MOVE_PENALTY = 1
GOAL_BONUS = 1000


def calculate_fitness(individual):
    result = simulate_individual(individual)

    # Mejor distancia a la meta alcanzada durante todo el recorrido.
    best_distance = min(
        BFS_DISTANCES[row][col]
        for row, col in result["path"]
    )

    if result["reached_goal"]:
        unused_steps = CHROMOSOME_LENGTH - result["steps_used"]

        return (
            GOAL_BONUS
            + unused_steps * 5
            - result["invalid_moves"] * INVALID_MOVE_PENALTY
        )

    progress = START_DISTANCE - best_distance

    return (
        progress * 10
        - result["invalid_moves"] * INVALID_MOVE_PENALTY
    )


calculate_fitness(example_individual)

POPULATION_SIZE = 60


def create_population():
    return [
        create_individual()
        for _ in range(POPULATION_SIZE)
    ]


population = create_population()

print("Individuos:", len(population))
print("Ejemplo:", population[0])

TOURNAMENT_SIZE = 3


def tournament_selection(population, fitness_values):
    candidate_indices = random.sample(
        range(len(population)),
        TOURNAMENT_SIZE
    )

    winner_index = max(
        candidate_indices,
        key=lambda index: fitness_values[index]
    )

    return population[winner_index]

def crossover(parent1, parent2):
    point = random.randint(1, CHROMOSOME_LENGTH - 1)

    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]

    return child1, child2

MUTATION_RATE = 0.05


def mutate(individual):
    mutated = individual.copy()

    for index, current_move in enumerate(mutated):
        if random.random() < MUTATION_RATE:
            alternatives = [
                move
                for move in MOVES
                if move != current_move
            ]
            mutated[index] = random.choice(alternatives)

    return mutated

GENERATIONS = 80
ELITE_SIZE = 4


def evolve(verbose=True):
    population = create_population()

    best_fitness_history = []
    mean_fitness_history = []

    best_individual = None
    best_fitness = float("-inf")

    for generation in range(GENERATIONS):
        fitness_values = [
            calculate_fitness(individual)
            for individual in population
        ]

        # Ordenamos índices de mejor a peor.
        ranked_indices = sorted(
            range(len(population)),
            key=lambda index: fitness_values[index],
            reverse=True,
        )

        generation_best_index = ranked_indices[0]
        generation_best_fitness = fitness_values[generation_best_index]

        if generation_best_fitness > best_fitness:
            best_fitness = generation_best_fitness
            best_individual = population[generation_best_index].copy()

        best_fitness_history.append(best_fitness)
        mean_fitness_history.append(
            sum(fitness_values) / len(fitness_values)
        )

        best_result = simulate_individual(best_individual)

        if verbose:
            print(
                f"Generación {generation:02d} | "
                f"Mejor fitness: {best_fitness:6.1f} | "
                f"Meta: {'sí' if best_result['reached_goal'] else 'no'}"
            )

        if best_result["reached_goal"]:
            return {
                "best_individual": best_individual,
                "best_fitness": best_fitness,
                "best_fitness_history": best_fitness_history,
                "mean_fitness_history": mean_fitness_history,
                "generations_used": generation + 1,
            }

        # Elitismo: conservar directamente los mejores.
        new_population = [
            population[index].copy()
            for index in ranked_indices[:ELITE_SIZE]
        ]

        # Crear descendencia hasta completar la población.
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(
                population,
                fitness_values
            )
            parent2 = tournament_selection(
                population,
                fitness_values
            )

            child1, child2 = crossover(parent1, parent2)

            new_population.append(mutate(child1))

            if len(new_population) < POPULATION_SIZE:
                new_population.append(mutate(child2))

        population = new_population

    return {
        "best_individual": best_individual,
        "best_fitness": best_fitness,
        "best_fitness_history": best_fitness_history,
        "mean_fitness_history": mean_fitness_history,
        "generations_used": GENERATIONS,
    }

random.seed(SEED)

result = evolve(verbose=True)

best_individual = result["best_individual"]
best_result = simulate_individual(best_individual)

print("\n--- Resultado ---")
print("Llegó a la meta:", best_result["reached_goal"])
print("Generaciones utilizadas:", result["generations_used"])
print("Movimientos utilizados:", best_result["steps_used"])
print("Fitness:", result["best_fitness"])
print("Cromosoma:", best_individual)

plot_maze(
    path=best_result["path"],
    title="Mejor recorrido encontrado por el algoritmo genético",
)

plt.figure(figsize=(8, 4))

plt.plot(
    result["best_fitness_history"],
    label="Mejor fitness"
)

plt.plot(
    result["mean_fitness_history"],
    label="Fitness promedio"
)

plt.xlabel("Generación")
plt.ylabel("Fitness")
plt.title("Evolución del fitness")
plt.legend()
plt.grid(alpha=0.3)
plt.show()