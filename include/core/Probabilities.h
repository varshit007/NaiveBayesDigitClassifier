//
// Created by varsh on 10/13/2020.
//
#pragma once
#include <fstream>
#include <unordered_map>
#include <vector>
#include "TrainingData.h"

class Probabilities {
 public:
  //Variables
  int num_of_training_images;
  size_t num_of_classes; 
  size_t image_length; 
  std::vector<double> class_prob;
  std::vector< std::vector< std::vector<double> > > shaded_pixel_prob;
  std::unordered_map <int, std::vector< std::vector< std::vector <bool> > > > images;
  
  std::vector <std::vector<std::vector<bool> > > test_images;
  std::vector<int> test_labels; 
  std::vector<int> classification_result; 
  
  //Methods
  Probabilities();
  Probabilities(naivebayes::TrainingData & training_data);
  void PixelShadedProb(); 
  void CalculateClassProbability();
  void FinalShadedPixelProb();
  int ClassifyImage(std::vector<std::vector<bool>> &);
  void ClassifyTestImages();
  friend std::ostream & operator << (std::ostream & out, Probabilities & probabilities);
  friend std::istream& operator>>(std::istream& is, Probabilities& probabilities);
  void GetTestImagesAndLabels(std::string test_labels_path, std::string test_images_path);
  void AddRow(std::string, std::vector<std::vector<bool>> &);
  double ClassificationAccuracy();
};
