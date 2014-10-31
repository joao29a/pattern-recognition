#include "MachineLearningWine.h"

int main() {
  MachineLearningWine ml;
  ml.constructDataCollection();
#ifdef PRINTDATA
  ml.printData();
#endif
  return 0;
}
