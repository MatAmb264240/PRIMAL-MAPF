#pragma once
#include <vector>
#include <set>
#include <cstdlib>
#include "TreeNode.h"
#include "util.h"

class LowLevelSolver
{
private:
    std::vector<std::vector<Cell>> optimalPaths;

    static inline bool isObstacle(const Map &map, int x, int y);
    static bool isConstraint(int agentID, int x, int y, int time, const std::vector<Constraint> &constraints);
    static bool contains(const std::vector<Cell> &cells, const Cell &cell);
    static int findIndex(const std::vector<Cell> &cells, const Cell &cell);
    static bool checkStartGoalCells(const Cell &start, const Cell &goal, const Map &map);
    static int findHeuristicDistance(const Cell &current_cell, const Cell &cell); // Manhattan distance calculation
    static bool isValid(int x, int y, const Map &map);                             // Checks if given cells are valid
    static int findMinCostIndex(const std::vector<Cell> &OPEN);
    friend std::ostream& operator<<(std::ostream& os, const LowLevelSolver& solver) {        
        for (const auto& row : solver.optimalPaths) {
            for (const auto& cell : row) {
                os << "(" << cell.x << ", " << cell.y << ") -> ";
            }
            os << "\n";
        }

        return os;
    }
public:
    LowLevelSolver();
    ~LowLevelSolver();

    // A* for single agent with constraints
    std::vector<Cell> solve(const std::vector<Constraint> &constraints, const Map &map, int agentID);

    // for each agent find optimal path
    std::vector<std::vector<Cell>> findOptimalPaths(const std::vector<Constraint> &constraints, const Map &map);

};
