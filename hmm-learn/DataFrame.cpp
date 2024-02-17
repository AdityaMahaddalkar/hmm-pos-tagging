#include "DataFrame.h"

Row::Row(int index, std::string word, std::string tag) {
	this->index = index;
	this->word = word;
	this->tag = tag;
}

DataFrame::DataFrame(std::vector<Row> rows) {
	this->rows = rows;
}