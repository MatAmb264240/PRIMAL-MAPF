#include "LowLevelSolver.h"
#include <algorithm>
#include <iostream>
#include <climits>
#include <limits>
#include <map>

LowLevelSolver::LowLevelSolver() = default;
LowLevelSolver::~LowLevelSolver() = default;

bool LowLevelSolver::checkStartGoalCells(const Cell &start, const Cell &goal, const Map &map)
{
    if (!isValid(start.x, start.y, map)) {
        std::cout<<"---------------------START on obstacle!!!" << start.x << ","<< start.y << ")\n";
        return false;
    }
    if (!isValid(goal.x, goal.y, map)) {
        std::cout<<"---------------------GOAL on obstacle!!!" << goal.x << ","<< goal.y << ")\n";
        return false;
    }
    if (start.isObstacle) {
        return false;
    }
    if (goal.isObstacle) {
        return false;
    }
    return true;
}

inline bool LowLevelSolver::isObstacle(const Map &map, int x, int y) {
    return map.cells[x][y].isObstacle;
}

inline bool LowLevelSolver::isValid(int x, int y, const Map &map) {
    if (x == 0 && y == 0) return true; 
    if (x < 0 || y < 0) return false;
    if (x >= static_cast<int>(map.cells.size())) return false;
    if (map.cells.empty()) return false;
    if (y >= static_cast<int>(map.cells[0].size())) return false;
    if (isObstacle(map, x, y)){
        return false;
    }
    return true;
}

// checks if the visited cell has a constraint at given time
bool LowLevelSolver::isConstraint(int agentID, int x, int y, int time, const std::vector<Constraint> &constraints) {
    for (const Constraint &c : constraints) {
        if (agentID == c.agentID) {
            if (time == c.time && c.cell.x == x && c.cell.y == y) {
                return true;
            }
        }
    }
    return false;
}

inline int LowLevelSolver::findHeuristicDistance(const Cell &current_cell, const Cell &goal)
{
    // Manhattan distance
    return std::abs(current_cell.x - goal.x) + std::abs(current_cell.y - goal.y);
}

bool LowLevelSolver::contains(const std::vector<Cell> &cells, const Cell &cell) {
    return std::find(cells.begin(), cells.end(), cell) != cells.end();
}

// returns index of cell in vector, or -1 if not found
int LowLevelSolver::findIndex(const std::vector<Cell> &cells, const Cell &cell) {
    auto it = std::find(cells.begin(), cells.end(), cell);
    if (it == cells.end()) return -1;
    return static_cast<int>(std::distance(cells.begin(), it));
}

// returns index of cell with minimal f value in OPEN
int LowLevelSolver::findMinCostIndex(const std::vector<Cell> &OPEN) {
    int minIndex = -1;
    int minValue = INT_MAX;

    for (int i = 0; i < static_cast<int>(OPEN.size()); ++i) {
        if (OPEN[i].f < minValue) {
            minValue = OPEN[i].f;
            minIndex = i;
        }
    }
    return minIndex;
}

// for each agent find optimal path
std::vector<std::vector<Cell>> LowLevelSolver::findOptimalPaths(const std::vector<Constraint> &constraints, const Map &map) {

    optimalPaths.clear();
    for (size_t k = 0; k < map.agents.size(); k++) {
        optimalPaths.emplace_back(solve(constraints, map, static_cast<int>(k)));
    }
    return optimalPaths;
}

// Classic A* with time dimension = g (step number) and constraints
std::vector<Cell> LowLevelSolver::solve(const std::vector<Constraint> &constraints, const Map &map, int agentID) {
    std::vector<Cell> OPEN;
    std::vector<Cell> CLOSED;

    std::vector<Cell> emptyResult;

    if (agentID < 0 || agentID >= static_cast<int>(map.agents.size())) {
        return emptyResult;
    }

    Cell start = map.agents[agentID].start;
    Cell goal = map.agents[agentID].end;

    if (!checkStartGoalCells(start, goal, map)) {
        return emptyResult;
    }

    // Initialize start node
    start.g = 0;
    start.h = findHeuristicDistance(start, goal);
    start.f = start.g + start.h;
    start.parent = nullptr;

    OPEN.push_back(start);

    // parent map: (x,y) -> (px,py)
    std::map<std::pair<int,int>, std::pair<int,int>> parent;
    parent[std::make_pair(start.x, start.y)] = std::make_pair(start.x, start.y);

    while (!OPEN.empty()) {
        int currentIndex = findMinCostIndex(OPEN);
        if (currentIndex == -1) {
            break;
        }

        Cell current = OPEN[currentIndex];
        OPEN.erase(OPEN.begin() + currentIndex);
        CLOSED.push_back(current);

        // goal check
        if (current == goal) {
            // reconstruct path
            std::vector<Cell> path;
            std::pair<int,int> curCoord = std::make_pair(current.x, current.y);

            while (true) {
                Cell c = map.cells[curCoord.first][curCoord.second];
                path.push_back(c);
                auto it = parent.find(curCoord);
                if (it == parent.end()) {
                    // should not happen, break to avoid infinite loop
                    break;
                }
                if (it->second == curCoord) {
                    // reached start
                    break;
                }
                curCoord = it->second;
            }

            std::reverse(path.begin(), path.end());
            return path;
        }

        // generate 4 neighbors
		// 5 actions: up, down, left, right, WAIT
		const int dx[5] = { -1, 1, 0, 0, 0 };
		const int dy[5] = { 0, 0, -1, 1, 0 };

		for (int dir = 0; dir < 5; ++dir) {
			int nx = current.x + dx[dir];
			int ny = current.y + dy[dir];

			// WAIT stays in place – must be valid coordinate, but don't treat it as obstacle check for move
			if (dir < 4 && !isValid(nx, ny, map)) {
				continue;
			}

			int nextTime = current.g + 1;

			if (isConstraint(agentID, nx, ny, nextTime, constraints)) {
				continue;
			}

			Cell neighbor = map.cells[nx][ny];
			neighbor.g = nextTime;
			neighbor.h = findHeuristicDistance(neighbor, goal);
			neighbor.f = neighbor.g + neighbor.h;

			int closedIndex = findIndex(CLOSED, neighbor);
			if (closedIndex != -1 && CLOSED[closedIndex].g <= neighbor.g) {
				continue;
			}

			int openIndex = findIndex(OPEN, neighbor);
			if (openIndex != -1 && OPEN[openIndex].g <= neighbor.g) {
				continue;
			}

			parent[{neighbor.x, neighbor.y}] = {current.x, current.y};

			if (openIndex != -1)
				OPEN[openIndex] = neighbor;
			else
				OPEN.push_back(neighbor);
		}

    }

    // no path found
    return emptyResult;
}
