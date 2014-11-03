#ifndef __FILEIO
#define __FILEIO

#include <fstream>
#include <unordered_map>
#include <vector>
#include "DataType.h"

template <typename T>
class FileIO {
  protected:
    std::ifstream ifs;
  public:
    FileIO(){}
    virtual void readFile(const char*, DataMap<T>&) {}
    virtual void readInput(const char*, DataMap<T>&) {}
};

#endif
