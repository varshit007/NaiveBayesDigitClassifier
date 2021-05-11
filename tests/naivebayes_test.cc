#include <core/TrainingData.h>
#include <core/Probabilities.h>
#include <catch2/catch.hpp>
#include <vector>
#include <fstream>
#include <iostream>
using std::vector;

TEST_CASE("Check if the training images are parsed correctly") {
  std::unordered_map <int, vector< vector< vector <bool> > > > images;

  vector<vector<bool>> first_1 = {  {false, true, true, true, false, false},
                                    {false, true, true, true, false, false},
                                    {false, true, true, true, false, false},
                                    {false, true, true, true, false, false},
                                    {false, true, true, true, false, false},
                                    {false, false, false, false, false, false}
                                  };

  vector<vector<bool>> second_1 = {   {false, false, true, false, false, false},
                                      {false, true, true, false, false, false},
                                      {false, false, true, false, false, false},
                                      {false, false, true, false, false, false},
                                      {true, true, true, true, true, false},
                                      {false, false, false, false, false, false}
                                  };

  vector<vector<bool>> third_1 = {    {false, false, true, false, false, false},
                                      {false, true, true, false, false, false},
                                      {false, true, true, false, false, false},
                                      {false, false, true, false, false, false},
                                      {true, true, true, true, true, false},
                                      {false, false, false, false, false, false}
                                  }; 

  images[1] = {first_1, second_1, third_1};

  naivebayes::TrainingData trainingData("C:\\Users\\varsh\\CLionProjects\\cinder_0.9.2_vc2015\\my-projects\\naivebayes-varshit007\\tests\\test_labels");
  std::ifstream stream ("C:\\Users\\varsh\\CLionProjects\\cinder_0.9.2_vc2015\\my-projects\\naivebayes-varshit007\\tests\\test_training_images");
  stream >> trainingData;
  stream.close(); 
  REQUIRE(images.at(1) == trainingData.images.at(1)); 
}


TEST_CASE("Calculate class probability") {
  std::ifstream in("C:\\Users\\varsh\\CLionProjects\\cinder_0.9.2_vc2015\\my-projects\\naivebayes-varshit007\\tests\\test_training_images"); 

  naivebayes::TrainingData training_data("C:\\Users\\varsh\\CLionProjects\\cinder_0.9.2_vc2015\\my-projects\\naivebayes-varshit007\\tests\\test_labels");

  in >> training_data;
  in.close();
  Probabilities probabilities(training_data);
  probabilities.CalculateClassProbability();
  
  double error = 0.0001; 
  REQUIRE(std::abs(0.0526316 - probabilities.class_prob.at(0)) < error);
  REQUIRE(std::abs(0.210526 - probabilities.class_prob.at(1)) < error);
  REQUIRE(std::abs(0.210526 - probabilities.class_prob.at(2)) < error);
  REQUIRE(std::abs(0.0526316 - probabilities.class_prob.at(6)) < error);
  
}


TEST_CASE("Calculate shaded/unshaded pixel probability") {
  std::ifstream in("C:\\Users\\varsh\\CLionProjects\\cinder_0.9.2_vc2015\\my-projects\\naivebayes-varshit007\\tests\\test_training_images");

  naivebayes::TrainingData training_data("C:\\Users\\varsh\\CLionProjects\\cinder_0.9.2_vc2015\\my-projects\\naivebayes-varshit007\\tests\\test_labels");

  in >> training_data;
  in.close();
  Probabilities probabilities(training_data);
  probabilities.PixelShadedProb();
  
  REQUIRE(0.5 == probabilities.shaded_pixel_prob.at(0).at(0).at(0));
  REQUIRE(0.2 == probabilities.shaded_pixel_prob.at(1).at(0).at(0));
  REQUIRE(0.8 == probabilities.shaded_pixel_prob.at(1).at(1).at(1));
  REQUIRE(0.6 == probabilities.shaded_pixel_prob.at(2).at(0).at(0));
}


TEST_CASE("Reading probs from a file") {
  
  SECTION("Reading shaded pixel probs from a file") {
    std::ifstream in(
        "C:\\Users\\varsh\\CLionProjects\\cinder_0.9.2_vc2015\\my-projects\\naivebayes-varshit007\\data\\mnistdatatraining\\trainingimages");

    naivebayes::TrainingData training_data(
        "C:\\Users\\varsh\\CLionProjects\\cinder_0.9.2_vc2015\\my-projects\\naivebayes-varshit007\\data\\mnistdatatraining\\traininglabels");

    in >> training_data;
    in.close();
    Probabilities probabilities(training_data);
    probabilities.CalculateClassProbability();
    probabilities.PixelShadedProb();

    std::ofstream os("C:\\Users\\varsh\\CLionProjects\\cinder_0.9.2_vc2015\\my-projects\\naivebayes-varshit007\\apps\\probs");
    os << probabilities;
    os.close();
    
    std::ifstream is(
        "C:\\Users\\varsh\\CLionProjects\\cinder_0.9.2_vc2015\\my-projects\\naivebayes-varshit007\\apps\\probs");
    Probabilities prob;
    is >> prob;
    is.close();
    REQUIRE(0.002079 == prob.shaded_pixel_prob.at(0).at(0).at(0));
    REQUIRE(0.00176991 == prob.shaded_pixel_prob.at(1).at(1).at(1));
    REQUIRE(0.00201207 == prob.shaded_pixel_prob.at(9).at(27).at(27));
  }

  SECTION("Reading class probs from a file") {
    std::ifstream is(
        "C:\\Users\\varsh\\CLionProjects\\cinder_0.9.2_vc2015\\my-projects\\naivebayes-varshit007\\apps\\probs");
    Probabilities probabilities;
    is >> probabilities;
    is.close();
    vector<double> class_prob = probabilities.class_prob;
    REQUIRE(0.0958084 == class_prob.at(0));
    REQUIRE(0.0868263 == class_prob.at(5));
    REQUIRE(0.099002 == class_prob.at(9));
  }
}


TEST_CASE("Test for classifying image") {
  SECTION("Classing the number 1") {
    std::ifstream in(
        "C:\\Users\\varsh\\CLionProjects\\cinder_0.9.2_vc2015\\my-projects\\naivebayes-varshit007\\tests\\test_training_images");

    naivebayes::TrainingData training_data(
        "C:\\Users\\varsh\\CLionProjects\\cinder_0.9.2_vc2015\\my-projects\\naivebayes-varshit007\\tests\\test_labels");

    in >> training_data;
    in.close();
    Probabilities probabilities(training_data);
    probabilities.CalculateClassProbability();
    probabilities.PixelShadedProb();

    vector<vector<bool>> image_1;
    image_1 = {{false, true, true, true, false, false},
             {false, true, true, true, false, false},
             {false, true, true, true, false, false},
             {false, true, true, true, false, false},
             {false, true, true, true, false, false},
             {false, false, false, false, false, false}};

    REQUIRE(1 == probabilities.ClassifyImage(image_1));
  }
  SECTION("Classing the number 3") {
    std::ifstream in(
        "C:\\Users\\varsh\\CLionProjects\\cinder_0.9.2_vc2015\\my-projects\\naivebayes-varshit007\\tests\\test_training_images");

    naivebayes::TrainingData training_data(
        "C:\\Users\\varsh\\CLionProjects\\cinder_0.9.2_vc2015\\my-projects\\naivebayes-varshit007\\tests\\test_labels");

    in >> training_data;
    in.close();
    Probabilities probabilities(training_data);
    probabilities.CalculateClassProbability();
    probabilities.PixelShadedProb();
    
    vector<vector<bool>> image_3;
    image_3 = {{true, true, true, true, false, false},
             {false, false, true, true, false, false},
             {true, true, true, true, false, false},
             {false, false, true, true, false, false},
             {true, true, true, true, false, false},
             {false, false, false, false, false, false}};

    REQUIRE(3 == probabilities.ClassifyImage(image_3));
  }
}

TEST_CASE("Check if the testing files are parsed correctly") {
  SECTION("Check if test labels are parsed correctly") {
    Probabilities probabilities;
    probabilities.GetTestImagesAndLabels("C:\\Users\\varsh\\CLionProjects\\cinder_0.9.2_vc2015\\my-projects\\naivebayes-varshit007\\data\\testlabels"
                                         , "C:\\Users\\varsh\\CLionProjects\\cinder_0.9.2_vc2015\\my-projects\\naivebayes-varshit007\\data\\testimages");
    REQUIRE(9 == probabilities.test_labels.at(0));
    REQUIRE(2 == probabilities.test_labels.at(50));
    REQUIRE(7 == probabilities.test_labels.at(500));
    REQUIRE(5 == probabilities.test_labels.at(999));
  }

  SECTION("Check if test images are parsed correctly") {
    Probabilities probabilities;
    probabilities.GetTestImagesAndLabels(
        "C:\\Users\\varsh\\CLionProjects\\cinder_0.9.2_vc2015\\my-projects\\naivebayes-varshit007\\data\\testlabels",
        "C:\\Users\\varsh\\CLionProjects\\cinder_0.9.2_vc2015\\my-projects\\naivebayes-varshit007\\data\\testimages");
    REQUIRE(true == probabilities.test_images.at(2).at(5).at(13));
    REQUIRE(false == probabilities.test_images.at(2).at(10).at(12));
  }
}

TEST_CASE("Classification Accuracy") {

  std::ifstream in(
      "C:\\Users\\varsh\\CLionProjects\\cinder_0.9.2_vc2015\\my-projects\\naivebayes-varshit007\\data\\mnistdatatraining\\trainingimages");

  naivebayes::TrainingData training_data(
      "C:\\Users\\varsh\\CLionProjects\\cinder_0.9.2_vc2015\\my-projects\\naivebayes-varshit007\\data\\mnistdatatraining\\traininglabels");

  in >> training_data;
  in.close();
  Probabilities probabilities(training_data);
  probabilities.CalculateClassProbability();
  probabilities.PixelShadedProb();

  
  probabilities.GetTestImagesAndLabels("C:\\Users\\varsh\\CLionProjects\\cinder_0.9.2_vc2015\\my-projects\\naivebayes-varshit007\\data\\testlabels"
      , "C:\\Users\\varsh\\CLionProjects\\cinder_0.9.2_vc2015\\my-projects\\naivebayes-varshit007\\data\\testimages");

  probabilities.ClassifyTestImages();
  REQUIRE(probabilities.ClassificationAccuracy() > 77.0);
}