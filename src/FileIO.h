#ifndef __FILEIO
#define __FILEIO

#include <fstream>
#include <unordered_map>
#include <vector>
#include "DataType.h"

template<typename T>
using DataMap = std::unordered_map<unsigned, std::vector<DataType<T>*>>;

template <typename T>
class FileIO {
  protected:
    std::ifstream ifs;
  public:
    FileIO(){}
    virtual bool readFile(const char*, DataMap<T>&) { return true; }
};

#endif
