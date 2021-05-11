#include <core/TrainingData.h>
#include <core/Probabilities.h>

#include <iostream>
#include <fstream>

int main(int argc, char * argv[]) {
  
  //Read in training data and training images and store them.
  std::ifstream in(
      "C:\\Users\\varsh\\CLionProjects\\cinder_0.9.2_vc2015\\my-projects\\naivebayes-varshit007\\data\\mnistdatatraining\\trainingimages");

  naivebayes::TrainingData training_data(
      "C:\\Users\\varsh\\CLionProjects\\cinder_0.9.2_vc2015\\my-projects\\naivebayes-varshit007\\data\\mnistdatatraining\\traininglabels");

  in >> training_data;
  in.close();
  Probabilities probabilities(training_data);
  probabilities.CalculateClassProbability();
  probabilities.PixelShadedProb();

  //Read in test labels and test images and store them.
  probabilities.GetTestImagesAndLabels("C:\\Users\\varsh\\CLionProjects\\cinder_0.9.2_vc2015\\my-projects\\naivebayes-varshit007\\data\\testlabels"
      , "C:\\Users\\varsh\\CLionProjects\\cinder_0.9.2_vc2015\\my-projects\\naivebayes-varshit007\\data\\testimages");

  probabilities.ClassifyTestImages();
  std::cout << "The classification accuracy is: " << probabilities.ClassificationAccuracy() << std::endl;
  return 0;
}
