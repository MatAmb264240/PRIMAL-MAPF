#include "TreeNode.h"
#include <climits>

TreeNode::TreeNode() = default;
TreeNode::~TreeNode() = default;

std::vector<std::vector<Cell>> TreeNode::getSolution() const
{
    return solution;
}

int TreeNode::getCost() const {
    return cost;
}

std::vector<Constraint> TreeNode::getConstraints() const {
    return constraints;
}

TreeNode::TreeNode(const std::vector<Constraint> &constraints) {
    this->constraints = constraints;
}

void TreeNode::addConstraint(const Constraint & constaint) {
    this->constraints.emplace_back(constaint);
}

void TreeNode::updateSolution(const Map &map) {
    LowLevelSolver solver;
    solution = solver.findOptimalPaths(constraints, map);
    std::cout<<solver;
}

void TreeNode::updateCost() {
    // if any agent has no path, this node is infeasible
    if (solution.empty()) {
        cost = INT_MAX;
        return;
    }

    for (const auto &route : solution) {
        if (route.empty()) {
            cost = INT_MAX;
            return;
        }
    }

    int totalCost = 0;
    for (const auto &route : solution) {
        totalCost += static_cast<int>(route.size());
    }
    this->cost = totalCost;
}

