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
    void readInput(const char*, DataMap<T>&) override;
};

template <typename T>
void WineIO<T>::readFile(const char* filename, DataMap<T>& data) {
  this->ifs.open(filename, std::ifstream::in);
  std::string line;
  unsigned id = 1, collection = 0;
  while (std::getline(this->ifs, line)) {
    std::istringstream ss(line);
    std::string word;
    std::getline(ss, word, ',');
    std::istringstream ss_val(word);
    unsigned val;
    ss_val >> val;
    if (val != collection) {
      collection = val;
      id = 1;
    }
    Wine<T>* wine = new Wine<T>();
    wine->setId(id);
    wine->setCollection(collection);
    data[collection].push_back(wine);
    for (std::string &att: wine->getAttributesName()) {
      assert(ss.good() && "No more words!");
      std::getline(ss, word, ',');
      std::istringstream ss_att(word);
      T valAtt;
      ss_att >> valAtt;
      wine->setAtt(att, valAtt);
    }
    id++;
  }
  this->ifs.close();
}

template <typename T>
void WineIO<T>::readInput(const char* input, DataMap<T>& data) {
  this->ifs.open(input, std::ifstream::in);
  std::string line;
  unsigned id = 1, collection = 0;
  while (std::getline(this->ifs, line)) {
    Wine<T>* wine = new Wine<T>();
    wine->setId(id);
    wine->setCollection(collection);
    data[collection].push_back(wine);
    std::istringstream ss(line);
    std::string word;
    for (std::string &att: wine->getAttributesName()) {
      std::getline(ss, word, ',');
      std::istringstream ssAtt(word);
      T valAtt;
      ssAtt >> valAtt;
      wine->setAtt(att, valAtt);
    }
    id++;
  }
  this->ifs.close();
}

#endif
