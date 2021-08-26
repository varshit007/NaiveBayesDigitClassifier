# NaiveBayesDigitClassifier
## Train, store, and load a model built from 5,000 training images of ASCII character digits into a Naïve Bayes Digit Classifier. This classifier classifies any given digit made of ASCII characters with 77% accuracy, and can be used to classify digits drawn by hand in an integrated GUI with slightly less accuracy.

![image](https://user-images.githubusercontent.com/47235519/130899834-3c670064-c2cb-4079-909a-987b13794cd0.png)
![image](https://user-images.githubusercontent.com/47235519/130899848-f619d7be-47fd-4d8b-ba2d-773c7d6dd2cd.png)

### What You Can Do:
- Draw and classify digits with the integrated GUI
- Train, load, and/or test the training model and/or classifier

### How To Run The Program:

#### Windows:
- Download Visual Studio Community 2015 with Update 3 at https://my.visualstudio.com/Downloads?q=visual%20studio%202015&wt.mc_id=o~msft~vscom~older-downloads
- When downloading, select custom installation and check the box for "Common Tools for Visual C++ 2015"
- Download Cinder for Visual C++ 2015 at https://libcinder.org/download
- Once these are both downloaded, open this repository in a code editor that supports CMake and build the project

#### To draw and classify digits in the integrated GUI:
- Select and run the "naive-bayes-gui" configuration, OR run the "naive-bayes-gui.exe" executable
- Note that the classification accuracy won't be the greatest, as the model was trained with digits made of ASCII characters

#### To train, load, and/or test the training model and/or classifier:
- Run the "train-load-test.exe" executable in a terminal, and enter the command line arguments as shown. For example, to train the model, type "train-load-test.exe train data/training_images.txt data/training_labels.txt data/training_model.txt"


##### Note: This program does not support MacOS, Linux, or Mobile.
