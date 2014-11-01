#ifndef __DATA_TYPE
#define __DATA_TYPE

#include <unordered_map>
#include <string>

template <typename T>
class DataType {
  protected:
    std::unordered_map<std::string, T> att;
    std::vector<std::string>* attributesName;

  public:
    DataType() { attributesName = nullptr; };
    T getAtt(std::string&);
    void setAtt(std::string&, T&);
    size_t getAttSize();
    std::vector<std::string>& getAttributesName() { return *attributesName; };
};

template <typename T>
void DataType<T>::setAtt(std::string &attName, T &data) {
  att[attName] = data;
}

template <typename T>
T DataType<T>::getAtt(std::string &attName) {
  return att[attName];
}

template <typename T>
size_t DataType<T>::getAttSize() {
  return attributesName->size();
}

#endif
