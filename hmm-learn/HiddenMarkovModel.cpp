#include "HiddenMarkovModel.h"
#include <iostream>

const std::string UNKNOWN_TOKEN = "< unk >";


size_t PairHash::operator()(const std::pair<std::string, std::string>& p) const {
	return std::hash<std::string>()(p.first) ^ (std::hash<std::string>()(p.second) << 1);
}

std::unordered_map<std::string, int> getCounts(const std::vector<std::string> &strings) {
	std::unordered_map<std::string, int> countMap;

	for (const std::string& s : strings) {
		if (countMap.find(s) != countMap.end()) {
			countMap[s]++;
		}
		else {
			countMap[s] = 1;
		}
	}

	return countMap;
}


HMM::HMM(const DataFrame &df) {

	std::vector<std::string> tagList;
	std::vector<std::string> wordList;


	// Get counts of words and tags
	for (const Row& row : df.rows) {

		tagList.push_back(row.tag);
		wordList.push_back(row.word);

		tagSet.insert(row.tag);
		wordSet.insert(row.word);

	}

	std::unordered_map<std::string, int> tagCountMap = getCounts(tagList);
	std::unordered_map<std::string, int> wordCountMap = getCounts(wordList);

	// Replace word with unknown whose count is below or equal to unkThreshold
	for (const Row &row : df.rows) {

		int wordCount = wordCountMap[row.word];

		if (wordCount <= unkThreshold) {
			wordCountMap.erase(row.word);
			wordSet.erase(row.word);
			
			if (wordCountMap.find(UNKNOWN_TOKEN) != wordCountMap.end()) {
				wordCountMap[UNKNOWN_TOKEN]++;
			}
			else {
				wordCountMap[UNKNOWN_TOKEN] = 1;
			}

			wordSet.insert(UNKNOWN_TOKEN);
		}
	}

	// Populate transition and emission map with counts;

	for (int index = 0;index < df.rows.size() - 2;index++) {

		std::string sourceTag = df.rows[index].tag;
		std::string destinationTag = df.rows[index + 1].tag;
		std::string emittedWord = df.rows[index].word;

		if (wordSet.find(emittedWord) == wordSet.end()) {
			emittedWord = UNKNOWN_TOKEN;
		}

		std::pair<std::string, std::string> transitionPair = std::make_pair(sourceTag, destinationTag);
		std::pair<std::string, std::string> emissionPair = std::make_pair(sourceTag, emittedWord);


		// Populate initial probabilities
		if (df.rows[index].index == 1) {
			if (initialProbabilities.find(sourceTag) != initialProbabilities.end()) {
				initialProbabilities[sourceTag]++;
			}
			else {
				initialProbabilities[sourceTag] = 1;
			}
		}

		// Populate transition and emission matrices

		if (transitions.find(transitionPair) != transitions.end()) {
			transitions[transitionPair]++;
		}
		else {
			transitions[transitionPair] = 1;
		}

		if (emissions.find(emissionPair) != emissions.end()) {
			emissions[emissionPair]++;
		}
		else {
			emissions[emissionPair] = 1;
		}
	}

	// Calculate transition and emission probabilities

	for (std::pair<const std::pair<std::string, std::string>, float>& transitionPair : transitions) {
		const std::string sourceTag = transitionPair.first.first;
		transitionPair.second /= tagCountMap[sourceTag]; //TODO: Check for no key found
	}

	for (std::pair<const std::pair<std::string, std::string>, float>& emissionPair : emissions) {
		const std::string tag = emissionPair.first.first;
		emissionPair.second /= tagCountMap[tag]; // TODO: Check for no key found
	}
}

std::vector<std::string> HMM::greedy(const DataFrame &df) {

	std::vector<std::string> tagList;
	std::string previousTag;

	for (const Row &row : df.rows) {

		std::string word;

		// Replace absent words with unk token
		if (wordSet.find(row.word) == wordSet.end()) {
			word = UNKNOWN_TOKEN;
		}
		else {
			word = row.word;
		}

		float maxProbability = -1;
		std::string bestTag;

		// Check for words at the start of the sentence
		if (row.index == 1) {

			for (const std::pair<std::string, float> &initialProbabilityPair : initialProbabilities) {

				std::string tag = initialProbabilityPair.first;
				std::pair<std::string, std::string> emissionPair = make_pair(tag, word);

				float initialProbability = initialProbabilityPair.second;
				float emissionProbability = emissions[emissionPair];

				if (initialProbability * emissionProbability > maxProbability) {
					maxProbability = initialProbability * emissionProbability;
					bestTag = tag;
				}

			}
		} // For other sentences
		else {

			for (const std::pair<std::pair<std::string, std::string>, float> &transitionPair : transitions) {

				
				std::string sourceTag = transitionPair.first.first;
				std::string destinationTag = transitionPair.first.second;
				std::pair<std::string, std::string> emissionPair = make_pair(destinationTag, word);

				if (sourceTag != previousTag) {
					continue;
				}

				float transitionProbability = transitionPair.second;
				float emissionProbability = emissions[emissionPair];

				if (transitionProbability * emissionProbability > maxProbability) {
					maxProbability = transitionProbability * emissionProbability;
					bestTag = destinationTag;
				}
			}
		}

		previousTag = bestTag;
		tagList.push_back(bestTag);
	}

	return tagList;
}

float HMM::accuracy_score(const DataFrame &df, const std::vector<std::string> &calculatedTags) {
	float accuracy = 0.0f;

	if (df.rows.size() != calculatedTags.size()) {
		std::cerr << "Size of dataframe and calculated tags do not match";
		exit(1);
	}

	for (int index = 0;index < calculatedTags.size();index++) {
		if (calculatedTags[index] == df.rows[index].tag) {
			accuracy += 1;
		}
	}

	return accuracy / calculatedTags.size();
}