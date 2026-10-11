"""Single entry point for EC Assignment 2 (Team 15).

Runs both tournament-size EA variants and the equal-budget random-search
baseline for 5 seeds, then writes plots and tables to __data__/experiments.

    uv run python main.py --time-eval   # estimate runtime first
    uv run python main.py --quick       # smoke test
    uv run python main.py               # full experiment

All options: uv run python main.py --help
"""

from experiments import main

if __name__ == "__main__":
    main()
