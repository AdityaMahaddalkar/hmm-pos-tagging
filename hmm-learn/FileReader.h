#pragma once

#ifndef FILE_READER_H
#define FILE_READER_H

#include <string>
#include <vector>
#include <iostream>
#include "DataFrame.h"


class FileReader {
public:
	DataFrame readFile(std::string filePath);
};

#endif // !FILE_READER_H
