#pragma once
#include "LowLevelSolver.h"
#include "util.h"

class TreeNode
{
private:
    int cost = 0;
    std::vector<Constraint> constraints;
    std::vector<std::vector<Cell>> solution;

public:
    void addConstraint(const Constraint &constaint);
    void updateSolution(const Map &map);
    void updateCost();

    std::vector<std::vector<Cell>> getSolution() const;
    int getCost() const;

    std::vector<Constraint> getConstraints() const;
    friend std::ostream& operator<<(std::ostream& os, const TreeNode& node) {
        os << "Cost: " << node.getCost() << "\n";
        os << "Constraints: " << node.constraints.size() << " constraints\n";
        os << "Solution: \n";
        
        for (const auto& row : node.getSolution()) {
            for (const auto& cell : row) {
                os << "(" << cell.x << ", " << cell.y << ") -> ";
            }
            os << "\n";
        }

        return os;
    }
    TreeNode();
    TreeNode(const std::vector<Constraint> &constraints);
    ~TreeNode();
};
