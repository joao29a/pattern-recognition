#ifndef __MACHINE_LEARNING
#define __MACHINE_LEARNING

#include "FileIO.h"
#include <iostream>

template<typename T>
class MachineLearning {
  protected:
    DataMap<T> collectionMap;
    FileIO<T>* dataIO;
    const char* filename;
  public:
    MachineLearning() {}
    void constructDataCollection();
    void printData();
};

template <typename T>
void MachineLearning<T>::constructDataCollection() {
  dataIO->readFile(filename, collectionMap);
}

template <typename T>
void MachineLearning<T>::printData() {
  for (auto& collection: collectionMap) {
    std::cout << "Collection " << collection.first << std::endl;
    for (DataType<T>* data: collection.second){
      for (std::string& att: data->getAttributesName())
        std::cout << "\t" << data->getAtt(att) << " ";
      std::cout << std::endl;
    }
  }
}

#endif
