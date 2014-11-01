#ifndef __WINEIO
#define __WINEIO

#include "FileIO.h"
#include "Wine.h"
#include <sstream>
#include <assert.h>

template<typename T>
class WineIO: public FileIO<T> {
  public:
    WineIO(){}
    void readFile(const char*, DataMap<T>&) override;
};

template <typename T>
void WineIO<T>::readFile(const char* filename, DataMap<T>& data) {
  this->ifs.open(filename, std::ifstream::in);
  std::string line;
  while (std::getline(this->ifs, line)) {
    std::istringstream ss(line);
    std::string word;
    std::getline(ss, word, ',');
    std::istringstream ss_val(word);
    unsigned val;
    ss_val >> val;
    Wine<T>* wine = new Wine<T>();
    data[val].push_back(wine);
    for (std::string &att: wine->getAttributesName()) {
      assert(ss.good() && "No more words!");
      std::getline(ss, word, ',');
      std::istringstream ss_att(word);
      T valAtt;
      ss_att >> valAtt;
      wine->setAtt(att, valAtt);
    }
  }
  this->ifs.close();
}

#endif
