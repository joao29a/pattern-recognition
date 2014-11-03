#ifndef __MACHINE_WINE
#define __MACHINE_WINE

#include "MachineLearning.h"
#include "WineIO.h"

#define TYPE double

class MachineLearningWine: public MachineLearning<TYPE> {
  public:
    MachineLearningWine();
};

MachineLearningWine::MachineLearningWine() : MachineLearning() {
  this->filename = "dataset-wine/wine.data";
  this->input    = "dataset-wine/input.data";
  this->dataIO   = new WineIO<TYPE>();
}

#endif
