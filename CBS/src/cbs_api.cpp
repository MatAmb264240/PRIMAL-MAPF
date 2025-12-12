#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <cstring>

#include "HighLevelSolver.h"
#include "LowLevelSolver.h"
#include "TreeNode.h"
#include "util.h"

// JSON lib — jeśli jeszcze jej nie masz:
// https://github.com/nlohmann/json
#include "json.hpp"
using json = nlohmann::json;

extern "C" {

/*
 Wejście (string JSON):
 {
   "grid_size": 16,
   "obstacles": [[1,2],[1,3],...],
   "starts":    [[x,y],[x,y],...],
   "goals":     [[x,y],[x,y],...]
 }

 Wyjście:
 {
   "paths": {
     "0": [[x,y],[x,y],...],
     "1": [[x,y],[x,y],...],
     ...
   }
 }
*/
char* solve_cbs(const char* json_input)
{
    // std::cout<<"-------- CPP ------------\n";
    try {
        json in = json::parse(json_input);
        
        int N = in["grid_size"];
        auto obstacles = in["obstacles"];
        auto starts    = in["starts"];
        auto goals     = in["goals"];

        Map map;
        map.cells.resize(N, std::vector<Cell>(N));
        
        for (int x = 0; x < N; x++) {
            for (int y = 0; y < N; y++) {
                map.cells[x][y] = Cell(x, y);
                map.cells[x][y].isObstacle = false;
            }
        }

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (obstacles[i][j] == 1) {  // Jeśli jest przeszkoda
                    map.cells[i][j].isObstacle = true;  // Ustawienie przeszkody w komórce (i, j)
                } else {  // Jeśli nie ma przeszkody
                    map.cells[i][j].isObstacle = false;  // Ustawienie wolnego miejsca
                }
            }
        }

        int num_agents = starts.size();
        map.agents.reserve(num_agents);
        
        for (int i = 0; i < num_agents; i++) {
            Agent ag(i);
            ag.start = Cell(starts[i][0], starts[i][1]);
            ag.end   = Cell(goals[i][0], goals[i][1]);
            map.agents.push_back(ag);
        }

        HighLevelSolver solver;
        auto solutionPaths = solver.solve(map);

        json out;
        json paths;


        for (size_t i = 0; i < solutionPaths.size(); i++) {
            json jpath = json::array();
            std::cout << "Agent " << i << " path: ";
            for (auto& c : solutionPaths[i]) {
                jpath.push_back({c.x, c.y});
                std::cout << "(" << c.x << ", " << c.y << ") ";
            }
            std::cout << std::endl;  // End of agent path
            paths[std::to_string(i)] = jpath;
        }

        out["paths"] = paths;

        std::string out_str = out.dump();
        char* result = (char*)malloc(out_str.size() + 1);
        std::memcpy(result, out_str.c_str(), out_str.size() + 1);

        return result;
    }
    catch (std::exception& e) {
        json out_err;
        out_err["error"] = e.what();
        std::string err = out_err.dump();

        char* result = (char*)malloc(err.size() + 1);
        std::memcpy(result, err.c_str(), err.size() + 1);

        // std::cout << "7 - Error occurred, returning error" << std::endl;
        return result;
    }
}

} // extern "C"
