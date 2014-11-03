#ifndef __MACHINE_LEARNING
#define __MACHINE_LEARNING

#include "FileIO.h"
#include <iostream>
#include <dlib/svm.h>

#define TOLERANCE  0.000000001
#define KERNEL_VAL 0.001
#define MAX_DICT   20000

std::ostream& operator<<(std::ostream&, const std::vector<unsigned>&);

template<typename T>
class MachineLearning {
  protected:
    typedef dlib::matrix<T, 0, 1> MatrixSample;
    typedef dlib::radial_basis_kernel<MatrixSample> Kernel;
    DataMap<T> collectionMap;
    DataMap<T> inputMap;
    FileIO<T>* dataIO;
    std::unordered_map<unsigned, dlib::kcentroid<Kernel>> learningData;
    const char* filename, *input;
    void trainData(std::vector<DataType<T>*>&, dlib::kcentroid<Kernel>&);
    size_t getAttSize();
    DataType<T>* getBestFromCollection(unsigned id, std::vector<DataType<T>*>, T*);
    DataType<T>* getWorstFromCollection(unsigned id, std::vector<DataType<T>*>, T*);
    std::vector<DataType<T>*> getAllSamples();
    void buildSamplesAndTrain(std::vector<DataType<T>*>, unsigned);
    MatrixSample makeSample(DataType<T>*);

  public:
    MachineLearning() {}
    void constructDataCollection();
    void buildLearningData();
    void testInputData();
    void printData();
    void printLearningInfo();
    void printBestAndWorst();
};

template <typename T>
void MachineLearning<T>::constructDataCollection() {
  dataIO->readFile(filename, collectionMap);
  dataIO->readInput(input, inputMap);
}

template <typename T>
void MachineLearning<T>::testInputData() {
  for (auto& learning: learningData) {
    DataType<T>* best, *worst;
    T distanceBest, distanceWorst;
    std::vector<unsigned> okSamples, badSamples, bestSamples;
    unsigned id = learning.first;
    if (id != collectionMap.size() + 1) {
      std::cout << "Collection: " << id << std::endl;
      DataType<T>* best = getBestFromCollection(id, collectionMap[id], 
          &distanceBest);
      DataType<T>* worst = getWorstFromCollection(id, collectionMap[id], 
          &distanceWorst);
    } else {
      std::cout << "Collection: All" << std::endl;
      DataType<T>* best = getBestFromCollection(id, getAllSamples(), 
          &distanceBest);
      DataType<T>* worst = getWorstFromCollection(id, getAllSamples(), 
          &distanceWorst);
    }
    dlib::kcentroid<Kernel> centroid = learning.second;
    for (auto& input: inputMap) {
      std::vector<DataType<T>*> samples = input.second;
      for (DataType<T>* sample: samples) {
        MatrixSample m = makeSample(sample);
        T distance = centroid(m);
        std::cout << "\tSample: " << sample->getId() << std::endl;
        std::cout << "\tAttributes:\n" << *sample;
        std::cout << "\tDistance: " << distance << std::endl;
        if (distance > distanceBest && distance < distanceWorst){
          std::cout << "\tResult: OK" << "\n\n";
          okSamples.push_back(sample->getId());
        }
        else if (distance <= distanceBest){
          std::cout << "\tResult: Best" << "\n\n";
          bestSamples.push_back(sample->getId());
        }
        else if (distance >= distanceWorst){
          std::cout << "\tResult: Bad" << "\n\n";
          badSamples.push_back(sample->getId());
        }
      }
    }
    std::cout << "\tStatistic: " << std::endl;
    std::cout << "\t\tOK Samples: " << okSamples << "(" << okSamples.size() 
      << ")" << std::endl;
    std::cout << "\t\tBest Samples: " << bestSamples << "(" << bestSamples.size() 
      << ")" << std::endl;
    std::cout << "\t\tBad Samples: " << badSamples << "(" << badSamples.size() 
      << ")" << "\n\n";
  }
}

template <typename T>
void MachineLearning<T>::buildLearningData() {
  for (auto& collection: collectionMap) {
    buildSamplesAndTrain(collection.second, collection.first);
  }
  if (collectionMap.size() > 1) {
    unsigned id = collectionMap.size() + 1;
    buildSamplesAndTrain(getAllSamples(), id);
  }
}

template <typename T>
void MachineLearning<T>::buildSamplesAndTrain(std::vector<DataType<T>*> samples,
    unsigned id) {
  dlib::kcentroid<Kernel> centroid(Kernel(KERNEL_VAL), TOLERANCE, MAX_DICT);
  trainData(samples, centroid);
  learningData[id] = centroid;
}

template <typename T>
void MachineLearning<T>::trainData(std::vector<DataType<T>*>& datas, 
    dlib::kcentroid<Kernel>& centroid) {
  for (DataType<T>* data: datas) {
    MatrixSample m = makeSample(data);
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

template <typename T>
std::vector<DataType<T>*> MachineLearning<T>::getAllSamples() {
  std::vector<DataType<T>*> samples;
  for (auto& collection: collectionMap) {
    for (DataType<T>* sample: collection.second)
      samples.push_back(sample);
  }
  return samples;
}

template <typename T>
typename MachineLearning<T>::MatrixSample MachineLearning<T>::makeSample(
    DataType<T>* sample) {
  MatrixSample m;
  m.set_size(getAttSize());
  for (size_t i = 0; i < getAttSize(); i++) {
    m(i) = sample->getAtt(sample->getAttributesName()[i]);
  }
  return m;
}

template <typename T>
DataType<T>* MachineLearning<T>::getBestFromCollection(unsigned id,
    std::vector<DataType<T>*> samples, T* distance) {
  T min = std::numeric_limits<T>::max();
  dlib::kcentroid<Kernel>& centroid = learningData[id];
  DataType<T>* best = nullptr;
  for (auto& sample: samples) {
    MatrixSample m = makeSample(sample);
    if (centroid(m) < min) {
      min = centroid(m);
      best = sample;
    }
  }
  if (distance != nullptr) *distance = min;
  return best;
}

template <typename T>
DataType<T>* MachineLearning<T>::getWorstFromCollection(unsigned id,
    std::vector<DataType<T>*> samples, T* distance) {
  T max = std::numeric_limits<T>::min();
  dlib::kcentroid<Kernel>& centroid = learningData[id];
  DataType<T>* worst = nullptr;
  for (auto& sample: samples) {
    MatrixSample m = makeSample(sample);
    if (centroid(m) > max) {
      max = centroid(m);
      worst = sample;
    }
  }
  if (distance != nullptr) *distance = max;
  return worst;
}

template <typename T>
void MachineLearning<T>::printBestAndWorst() {
  T distance;
  DataType<T>* bestSample = nullptr, *worstSample = nullptr;
  for (auto& collection: collectionMap) {
    std::cout << "Collection: " << collection.first << std::endl;
    std::cout << "\tTotal samples: " << collection.second.size() << "\n\n";
    bestSample = getBestFromCollection(collection.first, 
        collection.second, &distance);
    std::cout << "\tBest sample id: " << bestSample->getId() << std::endl;
    std::cout << "\tDistance: " << distance << "\n\n";
    worstSample = getWorstFromCollection(collection.first, 
        collection.second, &distance);
    std::cout << "\tWorst sample id: " << worstSample->getId() << std::endl;
    std::cout << "\tDistance: " << distance << "\n\n";
  }
  if (collectionMap.size() > 1) {
    std::cout << "All Collections " << std::endl;
    std::cout << "\tTotal samples: " << getAllSamples().size() << "\n\n";
    bestSample = getBestFromCollection(collectionMap.size() + 1, 
        getAllSamples(), &distance);
    std::cout << "\tBest sample id: " << bestSample->getId() << std::endl;
    std::cout << "\tCollection: " << bestSample->getCollection() << std::endl;
    std::cout << "\tDistance: " << distance << "\n\n";
    worstSample = getWorstFromCollection(collectionMap.size() + 1, 
      getAllSamples(), &distance);
    std::cout << "\tWorst sample id: " << worstSample->getId() << std::endl;
    std::cout << "\tCollection: " << worstSample->getCollection() << std::endl;
    std::cout << "\tDistance: " << distance << "\n\n";
  }
}

std::ostream& operator<<(std::ostream& os, const std::vector<unsigned>& vet) {
  for (unsigned val: vet) {
    os << val << " ";
  }
  return os;
}

#endif
