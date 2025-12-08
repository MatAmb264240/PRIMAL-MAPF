#include "HighLevelSolver.h"
#include <vector>
#include <algorithm>
#include <climits>

HighLevelSolver::HighLevelSolver() = default;
HighLevelSolver::~HighLevelSolver() = default;

// ------------------------------------------------------------
// Pomocniczo: pozycja agenta w czasie t, jeśli skończył ścieżkę
// wcześniej, to stoi na ostatniej komórce.
// ------------------------------------------------------------
static const Cell& getPositionAtTime(const std::vector<Cell>& route, size_t t)
{
    if (route.empty()) {
        // To nie powinno się zdarzać w poprawnym rozwiązaniu,
        // ale żeby nie było UB, zwracamy route.back()
        // – w praktyce lepiej upewnić się, że route nigdy nie jest puste.
        return *(&route.front()); // Undefined behavior jeśli route puste, ale zakładamy, że nie jest.
    }
    if (t < route.size())
        return route[t];
    return route.back();
}

// ------------------------------------------------------------
// Konflikt wierzchołkowy między dwiema trasami.
// Agent po końcu ścieżki stoi na swoim celu.
// ------------------------------------------------------------
bool HighLevelSolver::hasConflict(const std::vector<Cell> &route1,
                                  const std::vector<Cell> &route2)
{
    if (route1.empty() || route2.empty())
        return false;

    size_t max_index = std::max(route1.size(), route2.size());
    for (size_t t = 0; t < max_index; ++t) {
        const Cell& c1 = getPositionAtTime(route1, t);
        const Cell& c2 = getPositionAtTime(route2, t);
        if (c1 == c2)
            return true;
    }
    return false;
}

bool HighLevelSolver::hasConflict(const TreeNode &node)
{
    auto solutions = node.getSolution();
    for (size_t i = 0; i < solutions.size(); i++) {
        for (size_t j = i + 1; j < solutions.size(); j++) {
            if (hasConflict(solutions[i], solutions[j]))
                return true;
        }
    }
    return false;
}

// ------------------------------------------------------------
// Konflikt krawędziowy (swap) między dwiema trasami.
// Znowu: po końcu trasy agent stoi na celu.
// ------------------------------------------------------------
bool HighLevelSolver::hasEdgeConflict(const std::vector<Cell> &route1,
                                      const std::vector<Cell> &route2)
{
    if (route1.empty() || route2.empty())
        return false;

    // co najmniej 2 kroki, żeby mieć krawędzie
    size_t max_len = std::max(route1.size(), route2.size());
    if (max_len < 2) return false;

    for (size_t t = 1; t < max_len; ++t) {
        const Cell& prev1 = getPositionAtTime(route1, t - 1);
        const Cell& curr1 = getPositionAtTime(route1, t);
        const Cell& prev2 = getPositionAtTime(route2, t - 1);
        const Cell& curr2 = getPositionAtTime(route2, t);

        // swap: agent1 A->B, agent2 B->A
        if (prev1 == curr2 && curr1 == prev2)
            return true;
    }
    return false;
}

bool HighLevelSolver::hasEdgeConflict(const TreeNode &node)
{
    auto solutions = node.getSolution();
    for (size_t i = 0; i < solutions.size(); i++) {
        for (size_t j = i + 1; j < solutions.size(); j++) {
            if (hasEdgeConflict(solutions[i], solutions[j]))
                return true;
        }
    }
    return false;
}

// ------------------------------------------------------------
// Pierwszy konflikt (vertex albo edge), z logiką stania na celu.
// ------------------------------------------------------------
Conflict HighLevelSolver::getFirstConflict(const TreeNode &P)
{
    const auto &solutions = P.getSolution();
    int n = static_cast<int>(solutions.size());

    // 1. Szukamy konfliktów wierzchołkowych
    for (int i = 0; i < n; ++i) {
        const auto& r1 = solutions[i];
        for (int j = i + 1; j < n; ++j) {
            const auto& r2 = solutions[j];
            if (r1.empty() || r2.empty())
                continue;

            size_t max_len = std::max(r1.size(), r2.size());
            for (size_t t = 0; t < max_len; ++t) {
                const Cell& c1 = getPositionAtTime(r1, t);
                const Cell& c2 = getPositionAtTime(r2, t);
                if (c1 == c2) {
                    return Conflict(
                        i,
                        j,
                        c1,
                        c2,
                        static_cast<int>(t)
                    );
                }
            }
        }
    }

    // 2. Szukamy konfliktów krawędziowych (swap)
    for (int i = 0; i < n; ++i) {
        const auto& r1 = solutions[i];
        for (int j = i + 1; j < n; ++j) {
            const auto& r2 = solutions[j];
            if (r1.empty() || r2.empty())
                continue;

            size_t max_len = std::max(r1.size(), r2.size());
            if (max_len < 2) continue;

            for (size_t t = 1; t < max_len; ++t) {
                const Cell& prev1 = getPositionAtTime(r1, t - 1);
                const Cell& curr1 = getPositionAtTime(r1, t);
                const Cell& prev2 = getPositionAtTime(r2, t - 1);
                const Cell& curr2 = getPositionAtTime(r2, t);

                if (prev1 == curr2 && curr1 == prev2) {
                    return Conflict(
                        i,
                        j,
                        prev1,      // cell1 = z (t-1)
                        curr1,      // cell2 = do (t)
                        static_cast<int>(t)
                    );
                }
            }
        }
    }

    // Jeśli naprawdę nic nie ma (nie powinno się zdarzyć, gdy funkcja wołana
    // tylko przy hasConflict/hasEdgeConflict == true), zwróć coś neutralnego.
    return Conflict(0, 0, Cell(0,0), Cell(0,0), 0);
}

// ------------------------------------------------------------
// Reszta jak w Twojej wersji
// ------------------------------------------------------------

int HighLevelSolver::getMinCost(const std::vector<TreeNode> &tree)
{
    int min = INT_MAX;
    for (const auto &node : tree) {
        if (node.getCost() < min)
            min = node.getCost();
    }
    return min;
}

int HighLevelSolver::findBestNodeIndex(const std::vector<TreeNode> &tree)
{
    if (tree.empty()) return -1;
    int minCost = getMinCost(tree);
    for (int i = 0; i < static_cast<int>(tree.size()); ++i) {
        if (tree[i].getCost() == minCost)
            return i;
    }
    return -1;
}

inline bool HighLevelSolver::isEmpty(const std::vector<TreeNode> &tree)
{
    return tree.empty();
}

std::vector<std::vector<Cell>> HighLevelSolver::solve(const Map &map)
{
    for(int i=0; i<map.agents.size(); i++) {
        std::cout<< "Agent: " << i << "(" << map.agents[i].start.x << ", " << map.agents[i].start.y << ") -> " << "(" << map.agents[i].end.x << ", " << map.agents[i].end.y << ") \n";
    }
    std::vector<TreeNode> tree;

    TreeNode root;
    root.updateSolution(map);
    root.updateCost();

    if (root.getCost() < INT_MAX) {
        tree.emplace_back(root);
    }

    while (!isEmpty(tree)) {
        int bestIndex = findBestNodeIndex(tree);
        if (bestIndex == -1) {
            break;
        }

        TreeNode P = tree[bestIndex];
        tree.erase(tree.begin() + bestIndex);

        bool vertexConflict = hasConflict(P);
        bool edgeConflict   = hasEdgeConflict(P);

        if (!vertexConflict && !edgeConflict) {
            // konfliktów brak – zwracamy rozwiązanie
            return P.getSolution();
        }

        auto conflict = getFirstConflict(P);

        if (vertexConflict) {
            std::cout << "t\t";

            int a1 = conflict.conflictedAgentsID.first;
            int a2 = conflict.conflictedAgentsID.second;

            Cell c1 = conflict.cell1;
            Cell c2 = conflict.cell2;

            bool a1_on_goal = (map.agents[a1].end == P.getSolution()[a1].back());
            bool a2_on_goal = (map.agents[a2].end == P.getSolution()[a2].back());

            // --- PRZYPADKI ---

            // 1) Oba na celu → konflikt nierozwiązywalny → pomijamy węzeł
            if (a1_on_goal && a2_on_goal) {
                continue;
            }

            // 2) Agent1 na celu → dodajemy tylko ograniczenie dla agenta2
            if (a1_on_goal) {
                TreeNode A(P.getConstraints());
                A.addConstraint( Constraint(a2, c2, conflict.time) );
                A.updateSolution(map);
                A.updateCost();
                if (A.getCost() < INT_MAX)
                    tree.emplace_back(A);
                continue;
            }

            // 3) Agent2 na celu → dodajemy tylko ograniczenie dla agenta1
            if (a2_on_goal) {
                TreeNode A(P.getConstraints());
                A.addConstraint( Constraint(a1, c1, conflict.time) );
                A.updateSolution(map);
                A.updateCost();
                if (A.getCost() < INT_MAX)
                    tree.emplace_back(A);
                continue;
            }

            // 4) Normalny vertex conflict → dwie gałęzie
            {
                TreeNode A(P.getConstraints());
                A.addConstraint( Constraint(a1, c1, conflict.time) );
                A.updateSolution(map);
                A.updateCost();
                if (A.getCost() < INT_MAX)
                    tree.emplace_back(A);
            }
            {
                TreeNode A(P.getConstraints());
                A.addConstraint( Constraint(a2, c2, conflict.time) );
                A.updateSolution(map);
                A.updateCost();
                if (A.getCost() < INT_MAX)
                    tree.emplace_back(A);
            }
        }

        else if (edgeConflict) {
            // Edge conflict (swap): zabroń użycia tej krawędzi
            for (int side = 0; side < 2; ++side) {
                TreeNode A(P.getConstraints());

                int agentID = (side == 0
                               ? conflict.conflictedAgentsID.first
                               : conflict.conflictedAgentsID.second);

                Cell from = (side == 0 ? conflict.cell1 : conflict.cell2);
                Cell to   = (side == 0 ? conflict.cell2 : conflict.cell1);

                // Zakaz przechodzenia z "from" w t-1 do "to" w t
                Constraint c1(agentID, from, conflict.time - 1);
                Constraint c2(agentID, to,   conflict.time);

                A.addConstraint(c1);
                A.addConstraint(c2);
                A.updateSolution(map);
                A.updateCost();

                if (A.getCost() < INT_MAX) {
                    tree.emplace_back(A);
                }
            }
        }
    }

    // Brak rozwiązania – zwracamy pusty wektor
    return std::vector<std::vector<Cell>>();
}
