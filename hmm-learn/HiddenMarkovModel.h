#pragma once

#ifndef HIDDEN_MARKOV_MODEL_H
#define HIDDEN_MARKOV_MODEL_H

#include <unordered_map>
#include <unordered_set>
#include <string>
#include "DataFrame.h"


struct PairHash {
	size_t operator()(const std::pair<std::string, std::string>& p) const;
};

class HMM {
public:
	int unkThreshold = 2;

	std::unordered_set<std::string> tagSet;
	std::unordered_set<std::string> wordSet;

	std::unordered_map<std::string, float> initialProbabilities;
	std::unordered_map<std::pair<std::string, std::string>, float, PairHash> transitions;
	std::unordered_map<std::pair<std::string, std::string>, float, PairHash> emissions;

	HMM(const DataFrame &df);
	std::vector<std::string> greedy(const DataFrame &df);
	std::vector<std::string> viterbi(DataFrame df);
	float accuracy_score(const DataFrame &df, const std::vector<std::string> &calculatedTags);

};

#endif // !HIDDEN_MARKOV_MODEL_H
