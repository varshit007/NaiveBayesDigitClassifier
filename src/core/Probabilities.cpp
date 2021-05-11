//
// Created by varsh on 10/13/2020.
//
#include "../include/core/Probabilities.h"
#include <iostream>
#include <sstream>

/**
 * This is a default constructor which initializes few member variables.
 * This is only used while reading in the probabilities from a file.
 */
Probabilities::Probabilities() {
  this->num_of_classes = 10;
  this->image_length = 28;
}
/**
 * This is a custom constructor which creates the 'images' variable
 * @param training_data - We'll use this instance to update our images variable
 */
Probabilities::Probabilities(naivebayes::TrainingData & training_data) {
  this->images = training_data.images;
  this->num_of_training_images = training_data.labels.size();
  this->image_length = training_data.image_length;
  this->num_of_classes = 10;
  
  //Initializing every 'pixel prob' to 1 (the k value)
  shaded_pixel_prob.resize(num_of_classes);
  for (size_t i = 0; i < shaded_pixel_prob.size(); i++) {
    shaded_pixel_prob.at(i).resize(image_length);
    for (size_t j = 0; j < image_length; j++) {
      shaded_pixel_prob.at(i).at(j).resize(image_length, 1); 
    }
  }
  
}

/**
 * This is the operator overloading function (inserstion)
 * It write the class and shaded pixel probabilities to a file. 
 * @param out - The file where the probabilities will be written on. 
 * @param probabilities - The instance of the class which provides the data.
 * @return out
 */
std::ostream & operator << (std::ostream & out, Probabilities & probabilities) {
  out << probabilities.image_length;
  out << "\n";
  out << probabilities.num_of_classes; 
  out << "\n"; 
  
  for (size_t i = 0; i < probabilities.class_prob.size(); i++) {
    out << probabilities.class_prob.at(i); 
    out << "\n"; 
  }
  
  for (size_t i = 0; i < probabilities.shaded_pixel_prob.size(); i++) {
    for (size_t row = 0; row < probabilities.image_length; row++) {
      for (size_t col = 0; col < probabilities.image_length; col++) {
        out << probabilities.shaded_pixel_prob.at(i).at(row).at(col);
        out << " "; 
      }
      out << "\n"; 
    }
  }
  return out; 
}

/**
 * This method reads in a set of probabilities from a file (extraction)
 * It updates a few members varialbes like the class probs and pixel probs
 * @param is - File to be read from. 
 * @param probabilities - The instance of Probabilities that the changes are made to
 * @return is
 */
std::istream & operator>>(std::istream & is, Probabilities & probabilities) {
  //Getting the image length. 
  std::string length;
  std::getline(is, length);
  probabilities.image_length = std::stoi(length);

  std::string num_of_classes;
  std::getline(is, num_of_classes);
  probabilities.num_of_classes = std::stoi(num_of_classes);
  
  //Reading in class probabilities. 
  for(size_t i = 0; i < probabilities.num_of_classes; i++) {
    std::string single_class_prob;
    std::getline(is, single_class_prob);
    probabilities.class_prob.push_back( std::stod(single_class_prob) ); 
  }
  
  //Reading in shaded pixel probabilities
  std::string temp;
  
  std::vector<double> row; double prob; 
  std::vector<std::vector<double>> image_shaded_prob;
  
  for(auto class_prob : probabilities.class_prob) {
    for(size_t line_count = 0; line_count < probabilities.image_length; line_count++) {
      std::getline(is, temp);
      std::istringstream stream(temp);
      while(stream >> prob) {
        row.push_back(prob);
      }
      image_shaded_prob.push_back(row);
      row.clear();
    }
    probabilities.shaded_pixel_prob.push_back(image_shaded_prob);
    image_shaded_prob.clear();
  }
  return is; 
}

/**
 * This method calculates the probability that an image
 * in the training data belongs to a class c.
 */
void Probabilities::CalculateClassProbability() {
  double offset = 1.0;
  for (size_t i = 0; i <= 9; i++) {
    int num_of_class_images = images.at(i).size();
    double single_class_prob = (offset + num_of_class_images) / (10.0+num_of_training_images);
    class_prob.push_back(single_class_prob);
  }
}

/**
 * This function calculates the probability that a pixel is 
 * shaded given a class
 */
void Probabilities::PixelShadedProb() {
  for (size_t i = 0; i <= 9; i++) {
    for (size_t j = 0; j < images.at(i).size(); j++) {
      for (size_t row = 0; row < image_length; row++) {
        for (size_t col = 0; col < image_length; col++) {
          if(images.at(i).at(j).at(row).at(col) == true) {
            shaded_pixel_prob.at(i).at(row).at(col) += 1; 
          }
        }
      }
    }
  }
  FinalShadedPixelProb();
}

/**
 * This is just a helper methods to calculate shaded pixel probability
 * It updates the denominator of every pixel prob.
 */
void Probabilities::FinalShadedPixelProb() {
  for (size_t i = 0; i < shaded_pixel_prob.size(); i++) {
    for (size_t row = 0; row < image_length; row++) {
      for (size_t col = 0; col < image_length; col++) {
        int num_of_class_images = images.at(i).size();
        shaded_pixel_prob.at(i).at(row).at(col) /= (2+num_of_class_images); 
      }
    }
  }
}


/**
 * This function loads in the test images and test labels
 * and saves it to the test_images and test_labels vector.
 * @param test_labels_path - path to the test labels
 * @param test_images_path - path to the test images
 */
void Probabilities::GetTestImagesAndLabels(std::string test_labels_path, std::string test_images_path) {
  //Read in test labels
  std::ifstream stream(test_labels_path);
  std::string to_be_pushed;
  while (getline(stream, to_be_pushed)) {
    test_labels.push_back(std::stoi(to_be_pushed));
  }
  stream.close();

  //Read in test images as 3-D vectors
  std::ifstream images_stream(test_images_path);
  for (size_t i = 0; i < test_labels.size(); i++) {
    std::vector< std::vector <bool> > entire_image;

    for (size_t j = 0; j < image_length; j++) {
      std::string row;
      std::getline(images_stream, row);
      AddRow(row, entire_image);
    }
    test_images.push_back(entire_image);
  }
}

/**
 * This method takes in a row as a string,
 * converts it to a vector of bools and adds it to an image. 
 * @param row - The row which is to be added to the image.
 * @param entire_image - The image which is to be updated.  
 */
void Probabilities::AddRow(std::string row, std::vector<std::vector<bool>> & entire_image) {
  std::vector<bool> new_row;
  for (size_t i = 0; i < row.size(); i++) {
    if (row[i] != '\n') {
      if (row[i] == ' ') {
        new_row.push_back(false);
      } else {
        new_row.push_back(true);
      }
    }
  }
  entire_image.push_back(new_row);
}

/**
 * This image classifies one image
 * @param image - The image to be classified
 * @return The class that the image belongs to (0-9)
 */
int Probabilities::ClassifyImage(std::vector<std::vector<bool>> & image) {
  //Adding class probs to likelihood scores. 
  std::vector<double> likelihood; 
  for (size_t i = 0; i < class_prob.size(); i++) {
    likelihood.push_back(log(class_prob.at(i)));
  }
  //Adding shaded/unshaded probs to likelihood scores.
  for (size_t row = 0; row < image_length; row++) {
    for (size_t col = 0; col < image_length; col++) {
      size_t class_count = 0; 
      while (class_count < num_of_classes) {
        
        if (image.at(row).at(col) == true) {
          likelihood.at(class_count) += log (shaded_pixel_prob.at(class_count).at(row).at(col)); 
        } else if (image.at(row).at(col) == false) {
          likelihood.at(class_count) += log (1- shaded_pixel_prob.at(class_count).at(row).at(col));
        }
        class_count++;
        
      }
    }
  }
  //Max. likelihood score:
  double max_likelihood_score = likelihood.at(0);
  int max_likelihood_class = 0;
  for (size_t i = 1; i < likelihood.size(); i++) {
    if(max_likelihood_score < likelihood.at(i)) {
      max_likelihood_class = i; 
      max_likelihood_score = likelihood.at(i);
    }
  }
  //likelihood_scores.clear(); 
  return max_likelihood_class;
}

/**
 * This function classifies a list of images(test_images)
 * and updates the classification result.
 */
void Probabilities::ClassifyTestImages() {
  for (size_t i = 0; i < test_labels.size(); i++) {
    classification_result.push_back( ClassifyImage(test_images.at(i)) );
  }
}

/**
 * This fucntion calculates the classification accuracy. 
 * @return a double representing the accuracy if classified images.
 */
double Probabilities::ClassificationAccuracy() {
  double correct_count = 0; 
  for (size_t i = 0; i < classification_result.size(); i++) {
    if (test_labels.at(i) == classification_result.at(i)) {
      correct_count++; 
    }
  }
  return 100 * (correct_count / test_labels.size());
}