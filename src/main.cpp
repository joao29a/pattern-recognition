#include "MachineLearningWine.h"

int main(int argc, char** argv) {
  bool readFile = false;
  if (argc == 2) readFile = atoi(argv[1]);
  MachineLearningWine ml;
  ml.constructDataCollection();
  ml.buildLearningData(readFile);
#ifdef LEARNINGINFO
  ml.printLearningInfo();
#endif
#ifdef PRINTDATA
  ml.printData();
#endif
#ifdef BESTWORST
  ml.printBestAndWorst();
#endif
  ml.testInputData();
  return 0;
}
