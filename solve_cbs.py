# solve_cbs.py
import time
from master.agent import Agent
from master.map_handler import Map
from master.high_level_policy import HCBS


DEBUG = True          # <-- tu sterujesz
MAX_CBS_TIME = 5.0    # sekundy, zabezpieczenie


def _print_debug(grid_size, obstacles, starts, goals, solution=None, reason=""):
    print("\n================ CBS DEBUG ================")
    if reason:
        print("REASON:", reason)

    print("\nGRID:")
    for i in range(grid_size):
        row = ""
        for j in range(grid_size):
            if obstacles[i][j] == 1:
                row += "#"
            else:
                row += "."
        print(row)

    print("\nAGENTS:")
    for idx, (s, g) in enumerate(zip(starts, goals)):
        print(f"  Agent {idx}: start={s}, goal={g}")

    if solution is not None:
        print("\nPATHS:")
        for aid, path in solution.items():
            print(f"  Agent {aid}: {path}")

    print("==========================================\n")


def solve_cbs(grid_size, obstacles, starts, goals):
    Agent.id = 0

    grid = Map()
    grid.set_grid_cells(
        width=grid_size,
        height=grid_size,
        grid_cells=obstacles,
        diagonal_movements=False,
    )

    agents = []
    for (sx, sy), (gx, gy) in zip(starts, goals):
        agents.append(Agent(sx, sy, gx, gy))

    start_time = time.time()

    try:
        solution = HCBS(grid, agents, max_time=MAX_CBS_TIME)
    except Exception as e:
        if DEBUG:
            _print_debug(
                grid_size, obstacles, starts, goals,
                reason=f"EXCEPTION: {e}"
            )
        raise

    elapsed = time.time() - start_time

    if solution is None or solution is False:
        if DEBUG:
            _print_debug(
                grid_size, obstacles, starts, goals,
                reason="CBS returned None / False"
            )
        raise RuntimeError("CBS failed")

    if elapsed > MAX_CBS_TIME:
        if DEBUG:
            _print_debug(
                grid_size, obstacles, starts, goals,
                reason=f"TIMEOUT ({elapsed:.2f}s)"
            )
        raise RuntimeError("CBS timeout")

    # normalize output
    paths = {}
    for agent_id, (path, _) in solution.items():
        paths[agent_id] = [(n.i, n.j) for n in path]

    if DEBUG:
        # sanity check: no empty paths
        for aid, p in paths.items():
            if len(p) == 0:
                _print_debug(
                    grid_size, obstacles, starts, goals,
                    solution=paths,
                    reason=f"EMPTY PATH for agent {aid}"
                )
                raise RuntimeError("Empty path")

    return paths
