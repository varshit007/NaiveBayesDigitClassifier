#include "../include/core/TrainingData.h"
#include <fstream>

using std::vector;

namespace naivebayes {

/**
 * This is a custom constructor which reads in the 
 * labels from a given file and adds it to a 'labels' vector.
 * @param label_path - A path to the labels file. 
 */
TrainingData::TrainingData(std::string label_path) {
  std::ifstream stream(label_path);
  std::string to_be_pushed;
  while (getline(stream, to_be_pushed)) {
    labels.push_back(std::stoi(to_be_pushed));
  }
  
  //Initializing my map
  vector<vector<vector<bool>>> v;
  for (size_t i = 0; i <= 9; i++) {
    images.insert({i, v});
  }
}


/**
 * This is the operator overloading function. 
 * @param is - The file of training images that is to be read in. 
 * @param image - The instance of the class which is going to be modified. 
 * @return is
 */
std::istream& operator>>(std::istream& is, TrainingData& image) {
  //Getting the image length. 
  std::string row;    //One line of a file. 
  std::getline(is, row);
  size_t side_length = row.size();  //The size of the row will be the image length.
  image.image_length = side_length;

  is.seekg(0, std::ios_base::beg);  //We do this to go back to the first line of the file.
     
  //Maps a label of an image to a list of all those images.
  for (size_t i = 0; i < image.labels.size(); i++) {

    int label = image.labels.at(i);
    vector< vector <bool> > entire_image;

    for (size_t j = 0; j < image.image_length; j++) {
      std::getline(is, row);
      image.AddRow(row, entire_image);
    }

    image.images.at(label).push_back(entire_image);

  }
  return is;
}

/**
 * This method takes in a row as a string,
 * converts it to a vector of bools and adds it to an image. 
 * @param row - The row which is to be added to the image.
 * @param entire_image - The image which is to be updated.  
 */
void TrainingData::AddRow(std::string row, vector< vector <bool> > & entire_image ) {
  vector<bool> new_row; 
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

}  // namespace naivebayes