from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np

from ariel.ec import (
    EA,
    Crossover,
    EAOperation,
    FloatMutator,
    Individual,
    Population,
    set_seed
)
from simulation import evaluate as evaluate_genotype
from simulation import get_problem_size



DATA = Path.cwd() / "__data__" / "ea"

VARIANTS = {
    "low_pressure": 2,
    "high_pressure": 6,
}

_rng = np.random.default_rng()


@dataclass
class EAConfig:
    tournament_size: int = 2
    population_size: int = 30
    num_generations: int = 30
    n_elites: int = 1
    crossover_probability: float = 0.9
    mutation_sigma: float = 0.1
    mutation_probability: float = 1.0
    init_low: float = -1.0
    init_high: float = 1.0

    @property
    def n_offspring(self) -> int:
        return self.population_size - self.n_elites

    @property
    def evaluation_budget(self) -> int:
        return self.population_size + self.num_generations * self.n_offspring


def make_individual(n_genes: int, low: float, high: float) -> Individual:
    ind = Individual()
    ind.genotype = _rng.uniform(low, high, size=n_genes).tolist()
    return ind


def make_population(config: EAConfig) -> Population:
    n_genes = get_problem_size()[2]
    individuals = [
        make_individual(n_genes, config.init_low, config.init_high)
        for _ in range(config.population_size)
    ]
    return Population(individuals)

@EAOperation
def parent_selection(
    population: Population,
    tournament_size: int =2,
    n_offspring: int=29,
) -> Population:
    individuals = population.to_list()
    wins = [0] * len(individuals)
    for _ in range(n_offspring):
        contestants = _rng.choice(len(individuals), size=tournament_size, replace=False)
        best = min(contestants, key=lambda i: individuals[i].fitness)
        wins[best] += 1
    for ind, n in zip(individuals, wins):
        ind.tags = {"times_selected": n}
    return population

@EAOperation
def crossover(
    population: Population,
    crossover_probability: float = 0.9,
    n_offspring: int = 29,
) -> Population:
    pool = []
    for ind in population:
        times = cast("int", ind.tags.get("times_selected", 0))
        pool += [ind.genotype] * times
    _rng.shuffle(pool)
    children = []
    for i in range(0, len(pool), 2):
        a = pool[i]
        b = pool[(i + 1) % len(pool)]
        if _rng.random() < crossover_probability:
            a, b = Crossover.uniform(a, b)
        for genes in (a, b):
            if len(children) < n_offspring:
                child = Individual()
                child.genotype = list(genes)
                child.tags = {"is_child": True}
                children.append(child)
    population.extend(children)
    return population

@EAOperation
def mutate(
    population: Population,
    sigma: float = 0.1,
    mutation_probability: float = 1.0,
) -> Population:
    for ind in population:
        if ind.tags.get("is_child", False):
            genes = cast("list[float]", ind.genotype)
            ind.genotype = FloatMutator.gaussian(
                genes,
                std=sigma,
                mutation_probability=mutation_probability,
            )
            ind.requires_eval = True
    return population

@EAOperation
def evaluate(population: Population) -> Population:
    for ind in population.unevaluated:
        genes = cast("list[float]", ind.genotype)
        ind.fitness = evaluate_genotype(genes)
    return population

@EAOperation
def survivor_selection(population: Population, n_elites: int = 1) -> Population:
    parents = [ind for ind in population if not ind.tags.get("is_child", False)]
    children = [ind for ind in population if ind.tags.get("is_child", False)]
    parents.sort(key=lambda ind: ind.fitness)
    for ind in parents[n_elites:]:
        ind.alive = False
    for ind in children:
        ind.tags = {"is_child": False}
    return population

def statistics(population: Population, generation: int, evaluations: int) -> dict[str, float]:
    fitnesses = [ind.fitness for ind in population.alive]
    return {
        "generation": generation,
        "evaluations": evaluations,
        "best": min(fitnesses),
        "mean": sum(fitnesses) / len(fitnesses),
        "worst": max(fitnesses),
    }


def run_ea(
    variant: str = "low_pressure",
    seed: int = 42,
    config: EAConfig | None = None,
) -> dict[str, object]:
    global _rng
    config = replace(config or EAConfig(), tournament_size=VARIANTS[variant])
    _rng = np.random.default_rng(seed)
    set_seed(seed)

    population = make_population(config)
    evaluate(population)
    evaluations = config.population_size
    history = [statistics(population, 0, evaluations)]

    operations = [
        parent_selection(
            tournament_size=config.tournament_size,
            n_offspring=config.n_offspring,
        ),
        crossover(
            crossover_probability=config.crossover_probability,
            n_offspring=config.n_offspring,
        ),
        mutate(
            sigma=config.mutation_sigma,
            mutation_probability=config.mutation_probability,
        ),
        evaluate,
        survivor_selection(n_elites=config.n_elites),
    ]
    ea = EA(
        population,
        operations,
        num_steps=config.num_generations,
        is_maximisation=False,
        quiet=True,
        db_file_path=DATA / f"{variant}_seed{seed}.db",
        db_handling="delete",
    )
    for generation in range(1, config.num_generations + 1):
        ea.step()
        ea.fetch_population()
        evaluations += config.n_offspring
        history.append(statistics(ea.population, generation, evaluations))

    best = ea.get_solution("best", only_alive=False)
    return {
        "variant": variant,
        "seed": seed,
        "tournament_size": config.tournament_size,
        "history": history,
        "best_fitness": best.fitness,
        "best_genotype": best.genotype,
        "evaluations": evaluations,
    }


if __name__ == "__main__":
    test_config = EAConfig(population_size=10, num_generations=5)
    for name in VARIANTS:
        result = run_ea(name, seed=42, config=test_config)
        print(name, "k =", result["tournament_size"])
        for row in cast("list[dict[str, float]]", result["history"]):
            print(row)
        print("best:", result["best_fitness"], "evaluations:", result["evaluations"])