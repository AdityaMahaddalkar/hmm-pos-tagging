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

int main() {
	std::string trainFilePath = "C:\\Aditya\\Work\\USC\\anlp\\HW3\\data\\train";
	std::string testFilePath = "C:\\Aditya\\Work\\USC\\anlp\\HW3\\data\\test";
	std::string devFilePath = "C:\\Aditya\\Work\\USC\\anlp\\HW3\\data\\dev";

	FileReader fileReader;
	DataFrame train = fileReader.readFile(trainFilePath);
	DataFrame test = fileReader.readFile(testFilePath);
	DataFrame dev = fileReader.readFile(devFilePath);

	TimeVar hmmStart = timeNow();
	HMM hmm(train);
	TimeVar hmmEnd = timeNow();
	
	std::cout << "Took " << duration(hmmEnd - hmmStart) << " sec for HMM training\n";

	std::vector<std::string> testPredictions = runGreedyAndGetTestPredictions(hmm, train, dev, test);

}