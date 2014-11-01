#ifndef __MACHINE_LEARNING
#define __MACHINE_LEARNING

#include "FileIO.h"
#include <iostream>
#include <dlib/svm.h>
//#include <dlib/statistics.h>

#define TOLERANCE  0.0000001
#define KERNEL_VAL 0.1
#define MAX_DICT   100000

template<typename T>
class MachineLearning {
  protected:
    typedef dlib::matrix<T, 0, 1> MatrixSample;
    typedef dlib::radial_basis_kernel<MatrixSample> Kernel;
    typedef std::pair<MatrixSample, dlib::kcentroid<Kernel>> LearningPair;
    DataMap<T> collectionMap;
    FileIO<T>* dataIO;
    std::unordered_map<unsigned, LearningPair> learningData;
    const char* filename;
    void trainData(std::vector<DataType<T>*>&, MatrixSample&, 
                              dlib::kcentroid<Kernel>&);

  public:
    MachineLearning() {}
    void constructDataCollection();
    void buildLearningData();
    size_t getAttSize();
    void printData();
    void printLearningInfo();
};

template <typename T>
void MachineLearning<T>::constructDataCollection() {
  dataIO->readFile(filename, collectionMap);
}

template <typename T>
void MachineLearning<T>::buildLearningData() {
  MatrixSample mAll;
  mAll.set_size(getAttSize());
  dlib::kcentroid<Kernel> centroidAll(Kernel(KERNEL_VAL), TOLERANCE, MAX_DICT);
  bool hasCollections = false;
  if (collectionMap.size() > 1) hasCollections = true;
  for (auto& collection: collectionMap) {
    MatrixSample m;
    m.set_size(getAttSize());
    dlib::kcentroid<Kernel> centroid(Kernel(KERNEL_VAL), TOLERANCE, MAX_DICT);
    trainData(collection.second, m, centroid);
    LearningPair mlPair = {m, centroid};
    learningData[collection.first] = mlPair;
    if (hasCollections) trainData(collection.second, mAll, centroidAll);
  }
  if (hasCollections) {    
    LearningPair mlPair = {mAll, centroidAll};
    learningData[collectionMap.size() + 1] = mlPair;
  }
}

template <typename T>
void MachineLearning<T>::trainData(std::vector<DataType<T>*>& datas, 
    MatrixSample& m, dlib::kcentroid<Kernel>& centroid) {
  for (DataType<T>* data: datas) {
    for (size_t i = 0; i < getAttSize(); i++)
      m(i) = data->getAtt(data->getAttributesName()[i]);
    centroid.train(m);
  }
}

template <typename T>
size_t MachineLearning<T>::getAttSize() {
  return collectionMap.begin()->second.front()->getAttSize();
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

template <typename T>
void MachineLearning<T>::printLearningInfo() {
  for (auto& data: learningData) {
    std::cout << "Learning Data: " << data.first << std::endl;
    dlib::kcentroid<Kernel> centroid = data.second.second;
    std::cout << "\tSamples trained: " 
      << centroid.samples_trained() << std::endl;
    std::cout << "\tInner product: " 
      << centroid.inner_product(data.second.first) << std::endl;
    std::cout << "\tSquared norm: " 
      << centroid.squared_norm() << "\n\n";
  }
}

#endif
