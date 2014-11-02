#ifndef __DATA_TYPE
#define __DATA_TYPE

#include <unordered_map>
#include <string>

template <typename T>
class DataType;

template <typename T>
std::ostream& operator<<(std::ostream&, DataType<T>&);

template <typename T>
class DataType {
  protected:
    std::unordered_map<std::string, T> att;
    std::vector<std::string>* attributesName;
    unsigned id, collection;

  public:
    DataType() { attributesName = nullptr; };
    T getAtt(std::string&);
    void setAtt(std::string&, T&);
    size_t getAttSize();
    void setId(unsigned);
    unsigned getId();
    void setCollection(unsigned);
    unsigned getCollection();
    std::vector<std::string>& getAttributesName() { return *attributesName; };
};

template <typename T>
void DataType<T>::setAtt(std::string &attName, T &data) {
  att[attName] = data;
}

template <typename T>
void DataType<T>::setId(unsigned id) {
  this->id = id;
}

template <typename T>
unsigned DataType<T>::getId() {
  return id;
}

template <typename T>
void DataType<T>::setCollection(unsigned collection) {
  this->collection = collection;
}

template <typename T>
unsigned DataType<T>::getCollection() {
  return collection;
}

template <typename T>
T DataType<T>::getAtt(std::string &attName) {
  return att[attName];
}

template <typename T>
size_t DataType<T>::getAttSize() {
  return attributesName->size();
}

template <typename T>
std::ostream& operator<<(std::ostream& os, DataType<T>& dt) {
  for (std::string& att: dt.getAttributesName()){
    os << "\t\t" << att << ": " << dt.getAtt(att) << std::endl;
  }
  return os;
}

#endif
