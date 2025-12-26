import numpy as np
from collections import defaultdict
from master.node import GridNode
from master.open import GridOpen
from master.close import GridClose
import time

def calculate_cost(i1, j1, i2, j2):
    # Wait action costs 1
    return max(1, np.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2))


def diagonal_distance(i1, j1, i2, j2):
    d = 1
    d2 = np.sqrt(2)
    dx = abs(i1 - i2)
    dy = abs(j1 - j2)
    return d * (dx + dy) + (d2 - 2 * d) * min(dx, dy)


def manhattan_distance(i1, j1, i2, j2):
    return abs(i1 - i2) + abs(j1 - j2)


def AStar(grid_map, agent, constraints=set(), use_pc=False,
          heuristic_function=manhattan_distance,
          open_type=GridOpen, closed_type=GridClose,
          deadline=None):

    def make_path(goal):
        current = goal
        length = current.g
        path = []
        while current:
            path.append(current)
            current = current.parent
        return path[::-1], length

    if deadline is None:
        deadline = float('inf')

    OPEN = open_type()
    CLOSED = closed_type()
    best_time = dict()

    widths = defaultdict(int)
    opt_cost = float('inf')
    first_found_path = None

    max_constrain_t = max(constraints, key=lambda x: x[-1], default=[0])[-1]
    MAX_T = max_constrain_t + grid_map.height * grid_map.width

    goal = GridNode(agent.goal_i, agent.goal_j, t=-1)
    OPEN.add_node(GridNode(agent.start_i, agent.start_j, t=0, g=0, h=0))

    while len(OPEN) != 0:

        if time.time() > deadline:
            return None, float('inf')

        state = OPEN.get_best_node()

        key = (state.i, state.j)
        if key in best_time and best_time[key] <= state.t:
            continue
        best_time[key] = state.t

        CLOSED.add_node(state)

        if state == goal and state.t > max_constrain_t:
            if use_pc:
                if state.g <= opt_cost:
                    path, opt_cost = make_path(state)
                    if not first_found_path:
                        first_found_path = path
                    DAG = defaultdict(set)
                    OPEN = GridOpen()
                    OPEN.add_node(GridNode(agent.start_i, agent.start_j, t=0, g=0, h=0))
                    nodes_levels = []
                    while len(OPEN) != 0:
                        if time.time() > deadline:
                            return None, float('inf')
                        state = OPEN.get_best_node()
                        if state.f > opt_cost:
                            continue
                        if state == goal and state.g == opt_cost:
                            nodes_levels.insert(0, DAG[(state.i, state.j, state.t)])
                            widths[state.t - 1] += len(DAG[(state.i, state.j, state.t)])
                            while True:
                                w = set()
                                for s in nodes_levels[0]:
                                    w = w.union(DAG[s])
                                if not w:
                                    break
                                nodes_levels.insert(0, w)
                                widths[s[-1] - 1] += len(w)

                        next_coords = grid_map.get_neighbors(state.i, state.j)
                        for next_coord in next_coords:
                            if state.t + 1 > MAX_T:
                                continue
                            next_g = state.g + calculate_cost(state.i, state.j, next_coord[0], next_coord[1])
                            heuristic_dist = heuristic_function(next_coord[0], next_coord[1], goal.i, goal.j)
                            next_state = GridNode(next_coord[0], next_coord[1],
                                                  g=next_g, t=state.t + 1,
                                                  h=heuristic_dist, parent=state)
                            if (agent.id, next_state.i, next_state.j, next_state.t) in constraints \
                               or (agent.id, state.i, state.j,
                                   next_state.i, next_state.j, next_state.t) in constraints:
                                continue
                            DAG[(next_state.i, next_state.j, next_state.t)].add((state.i, state.j, state.t))
                            OPEN.add_node(next_state)
                    return first_found_path, opt_cost, widths
            else:
                return make_path(state)

        next_coords = grid_map.get_neighbors(state.i, state.j)
        for next_coord in next_coords:
            if state.t + 1 > MAX_T:
                continue
            next_g = state.g + calculate_cost(state.i, state.j, next_coord[0], next_coord[1])
            heuristic_dist = heuristic_function(next_coord[0], next_coord[1], goal.i, goal.j)
            next_state = GridNode(next_coord[0], next_coord[1],
                                  g=next_g, t=state.t + 1,
                                  h=heuristic_dist, parent=state)
            if CLOSED.was_expanded(next_state) \
               or (agent.id, next_state.i, next_state.j, next_state.t) in constraints \
               or (agent.id, state.i, state.j,
                   next_state.i, next_state.j, next_state.t) in constraints:
                continue
            OPEN.add_node(next_state)

    return None, float('inf')