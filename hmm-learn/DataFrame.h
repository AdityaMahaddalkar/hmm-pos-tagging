#pragma once

#ifndef DATA_FRAME_H
#define DATA_FRAME_H

#include <string>
#include <vector>

class Row {
public:
	int index;
	std::string word;
	std::string tag;
	Row(int index, std::string word, std::string tag);
};

class DataFrame {
public:
	std::vector<Row> rows;
	DataFrame(std::vector<Row> rows);
};

#endif // !DATA_FRAME_H
