#include <string>
#include <vector>
#include "FileReader.h"
#include "DataFrame.h"
#include "HiddenMarkovModel.h"
#include <chrono>

typedef std::chrono::high_resolution_clock::time_point TimeVar;

#define duration(a) std::chrono::duration_cast<std::chrono::seconds>(a).count()
#define timeNow() std::chrono::high_resolution_clock::now()

std::vector<std::string> runGreedyAndGetTestPredictions(HMM& hmm, const DataFrame& train, const DataFrame& dev, const DataFrame& test) {

	TimeVar greedyStart = timeNow();
	std::vector<std::string> trainTags = hmm.greedy(train);
	TimeVar greedyEnd = timeNow();
	std::cout << "Took " << duration(greedyEnd - greedyStart) << " sec for HMM prediction on train\n";
	std::cout << "Greedy accuracy for train: " << hmm.accuracy_score(train, trainTags) * 100 << "%\n";

	greedyStart = timeNow();
	std::vector<std::string> devTags = hmm.greedy(dev);
	greedyEnd = timeNow();
	std::cout << "Took " << duration(greedyEnd - greedyStart) << " sec for HMM prediction on dev\n";
	std::cout << "Greedy accuracy for dev: " << hmm.accuracy_score(dev, devTags) * 100 << "%\n";

	greedyStart = timeNow();
	std::vector<std::string> testTags = hmm.greedy(test);
	greedyEnd = timeNow();
	std::cout << "Took " << duration(greedyEnd - greedyStart) << " sec for HMM prediction on test\n";

	return testTags;
	
}

std::vector<std::string> runViterbiAndGetTestPredictions(HMM& hmm, const DataFrame& train, const DataFrame& dev, const DataFrame& test) {

	TimeVar viterbiStart = timeNow();
	std::vector<std::string> trainTags = hmm.viterbi(train);
	TimeVar viterbiEnd = timeNow();
	std::cout << "Took " << duration(viterbiEnd - viterbiStart) << " sec for HMM prediction on train\n";
	std::cout << "Viterbi accuracy for train: " << hmm.accuracy_score(train, trainTags) * 100 << "%\n";

	viterbiStart = timeNow();
	std::vector<std::string> devTags = hmm.viterbi(dev);
	viterbiEnd = timeNow();
	std::cout << "Took " << duration(viterbiEnd - viterbiStart) << " sec for HMM prediction on dev\n";
	std::cout << "Viterbi accuracy for dev: " << hmm.accuracy_score(dev, devTags) * 100 << "%\n";

	viterbiStart = timeNow();
	std::vector<std::string> testTags = hmm.viterbi(test);
	viterbiEnd = timeNow();
	std::cout << "Took " << duration(viterbiEnd - viterbiStart) << " sec for HMM prediction on test\n";

	return testTags;

}

int main() {
	std::string trainFilePath = "train_path";
	std::string testFilePath = "test_path";
	std::string devFilePath = "dev_path";

	FileReader fileReader;
	DataFrame train = fileReader.readFile(trainFilePath);
	DataFrame test = fileReader.readFile(testFilePath);
	DataFrame dev = fileReader.readFile(devFilePath);

	TimeVar hmmStart = timeNow();
	HMM hmm(train);
	TimeVar hmmEnd = timeNow();
	
	std::cout << "Took " << duration(hmmEnd - hmmStart) << " sec for HMM training\n";

	std::vector<std::string> testPredictionsGreedy = runGreedyAndGetTestPredictions(hmm, train, dev, test);
	std::vector<std::string> testPredictionsViterbi = runViterbiAndGetTestPredictions(hmm, train, dev, test);

}