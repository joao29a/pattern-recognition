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
  return 0;
}
