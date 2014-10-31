#ifndef __WINE
#define __WINE

#include "DataType.h"
#include <vector>

template <typename T>
class Wine: public DataType<T> {
  public:
    Wine();
};

template <typename T>
Wine<T>::Wine() : DataType<T>() {
  this->attributesName = new std::vector<std::string> {"alcohol", "malic acid", 
      "ash", "alcalinity of ash", "magnesium", "total phenols", "flavanois", 
      "nonflavanoid phenols", "roanthocyanins", "color intensity", "hue", 
      "OD280/OD315 of diluted wines", "proline"};
}

#endif
