#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include "FileReader.h"


DataFrame FileReader::readFile(std::string filePath) {
	std::ifstream inFile(filePath);

	if (inFile.is_open()) {

		std::vector<Row> rows;

		// Read and process each line of the file
		std::string line;
		while (std::getline(inFile, line)) {

			if (line.empty()) {
				continue;
			}

			// Create a stringstream to parse the line
			std::stringstream ss(line);

			// Variables to store the values
			int index;
			std::string word, tag;

			// Read the values from the stringstream
			ss >> index >> word >> tag;

			rows.push_back(Row(index, word, tag));

		}

		inFile.close();

		return DataFrame(rows);

	}
	else {
		std::cerr << "No such file found";
		exit(1);
	}
}
