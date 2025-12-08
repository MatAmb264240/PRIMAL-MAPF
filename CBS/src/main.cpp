// ConflictBasedSearch.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include <chrono>
#include "../lib/LowLevelSolver.h"
#include "../lib/HighLevelSolver.h"

std::vector<std::string> split(std::string s, char delim) {
	std::vector<std::string> result;
	std::stringstream ss(s);
	std::string item;

	while (getline(ss, item, delim)) {
		result.push_back(item);
	}
	return result;
}

Map readMap(std::string filename) {

	Map map; // Grid map
	int row, col;

	std::ifstream infile(filename);
	std::string line;

	getline(infile, line);

	auto splitted = split(line, ' ');
	row = std::stoi(splitted[0]);
	col = std::stoi(splitted[1]);

	getline(infile, line);

	// split second line to get obstacles
	std::istringstream k(line);
	std::string s;

	std::vector<std::vector<Cell>> cells;

	for (int i = 0; i < row; i++) {

		std::vector<Cell> row;
		for (int j = 0; j < col; j++) {
			row.emplace_back(Cell(i, j));
		}
		cells.emplace_back(row);
	};

	while (getline(k, s, ' ')) {

		int obstacleIndex = std::stoi(s);
		int x = obstacleIndex % row;
		int y = obstacleIndex / row;

		cells[x][y].isObstacle = true;
	}

	// next lines are agents. startIndex endIndex
	int start, end;

	std::vector<Agent> agents;

	int agentID = 0;
	while (infile >> start >> end) {

		int startX = start % row;
		int startY = start / row;

		int endX = end % row;
		int endY = end / row;

		Agent agent(agentID);
		agent.start = Cell(startX, startY);
		agent.end = Cell(endX, endY);
		std::cout << "start: "  << startX << startY;
		std::cout << " end: " << endX << endY << "\n";
		agents.emplace_back(agent);
		agentID++;
	}

	map.cells = cells;
	map.agents = agents;
	return map;
}

void printMap(Map map) {
	for (int i = 0; i < map.cells.size(); i++) {
		for (int j = 0; j < map.cells[0].size(); j++) {
			if (map.cells[j][i].isObstacle)
				std::cout << "X";
			else
				std::cout << "_";
		}
		std::cout << std::endl;
	}
}

void printAgents(Map map) {
	for (int i = 0; i < map.agents.size(); i++) {
		std::cout << map.agents[i].start.y;

	}
}

void testLowLevel(Map map) {
	LowLevelSolver solver;
	auto x = solver.findOptimalPaths(std::vector<Constraint>(), map);

	std::cout << x.size();
	for (int i = 0; i < x.size(); i++) {
		for (int j = 0; j < x[0].size(); j++) {
			std::cout << x[i][j].x << "," << x[i][j].y << std::endl;
		}
	}

}

unsigned int Factorial(unsigned int number) {
	return number <= 1 ? number : Factorial(number - 1)*number;
}

void printSolution(std::vector<std::vector<Cell>> optimalPaths) {

	for (auto path : optimalPaths) {
		std::cout << "Optimal path of agent \n";
		for (auto cell : path) {
			std::cout << cell.x << cell.y << "\n";
		}
	}
}


// int main() {
//     auto started = std::chrono::high_resolution_clock::now();

// 	std::vector<std::vector<Cell>> optimalPaths;
// 	Map map = readMap("data\\map3.txt");
// 	printMap(map);
// 	HighLevelSolver solver;
// 	optimalPaths = solver.solve(map);
// 	printSolution(optimalPaths);

//     auto done = std::chrono::high_resolution_clock::now();
//     auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(done-started).count();

//     std::cout << elapsedTime << " milliseconds " ;

//     return 0;
// }


int main() {
    // Definiowanie mapy 3x3 bez przeszkód
    int mapWidth = 3;
    int mapHeight = 3;
    
    Map map;
    map.cells.resize(mapWidth, std::vector<Cell>(mapHeight, Cell(0, 0)));
    
    // Ustawianie pozycji dla każdej komórki mapy
    for (int i = 0; i < mapWidth; ++i) {
        for (int j = 0; j < mapHeight; ++j) {
            map.cells[i][j] = Cell(i, j);
        }
    }

    // Definiowanie pozycji startowych i celowych dla dwóch agentów
	Cell start(0,0);
	Cell end(2,2);
    Agent agent1(0);
	agent1.start=start;
	agent1.end=end;
    map.agents.push_back(agent1);

	Cell start2(2,0);
	Cell end2(0,2);
    Agent agent2(1);
	agent2.start=start2;
	agent2.end=end2;
    map.agents.push_back(agent2);
    std::vector<Constraint> constraints;


    // constraints.push_back(Constraint(0, Cell(1, 0), 1));
    // constraints.push_back(Constraint(0, Cell(1, 0), 2));

    // constraints.push_back(Constraint(0, Cell(0, 1), 1)); 
    //constraints.push_back(Constraint(0, Cell(0, 1), 2)); 

    LowLevelSolver lowLevel;


    std::vector<Cell> pathAgent1 = lowLevel.solve(constraints, map, 0);
    
    if (pathAgent1.empty()) {
        std::cout << "Nie znaleziono ścieżki dla agenta 1." << std::endl;
    } else {
        std::cout << "Znaleziono ścieżkę dla agenta 1:" << std::endl;
        for (const Cell& cell : pathAgent1) {
            std::cout << "(" << cell.x << ", " << cell.y << ")" << std::endl;
        }
    }
	std::cout<<"\n\n---------2222222-------------\n";
	std::vector<Cell> pathAgent2 = lowLevel.solve(constraints, map, 1);
    
    if (pathAgent1.empty()) {
        std::cout << "Nie znaleziono ścieżki dla agenta 1." << std::endl;
    } else {
        std::cout << "Znaleziono ścieżkę dla agenta 1:" << std::endl;
        for (const Cell& cell : pathAgent1) {
            std::cout << "(" << cell.x << ", " << cell.y << ")" << std::endl;
        }
    }


    return 0;
}