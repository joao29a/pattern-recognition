#include "MachineLearningWine.h"

int main() {
  MachineLearningWine ml;
  ml.constructDataCollection();
  ml.buildLearningData();
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
