#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace naivebayes {

class TrainingData {
 public:
  //Variables
  size_t image_length;
  std::unordered_map <int, std::vector< std::vector< std::vector <bool> > > > images;
  std::vector<int> labels;
  //Methods
  TrainingData(std::string path);
  friend std::istream& operator>>(std::istream& is, TrainingData& td);
  void AddRow(std::string, std::vector< std::vector <bool> > & entire_image);
};

}  // namespace naivebayes