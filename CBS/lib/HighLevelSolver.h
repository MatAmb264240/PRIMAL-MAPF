#pragma once
#include "TreeNode.h"
#include "util.h"

class HighLevelSolver
{
private:
    static inline bool isEmpty(const std::vector<TreeNode> &tree);

    // Konflikt wierzchołkowy między dwoma trasami
    static bool hasConflict(const std::vector<Cell> &route1,
                            const std::vector<Cell> &route2);

    // Konflikt wierzchołkowy w całym węźle drzewa
    static bool hasConflict(const TreeNode &P);

    // Konflikt krawędziowy (swap) między dwoma trasami
    static bool hasEdgeConflict(const std::vector<Cell> &route1,
                                const std::vector<Cell> &route2);

    // Konflikt krawędziowy w całym węźle drzewa
    static bool hasEdgeConflict(const TreeNode &P);

    static int getMinCost(const std::vector<TreeNode> &tree);

    // Zwraca pierwszy konflikt (wierzchołek lub krawędź),
    // z logiką "agent po końcu ścieżki stoi na celu"
    static Conflict getFirstConflict(const TreeNode &P);

    static int findBestNodeIndex(const std::vector<TreeNode> &tree);

public:
    HighLevelSolver();
    ~HighLevelSolver();
    static std::vector<std::vector<Cell>> solve(const Map &map);
};
