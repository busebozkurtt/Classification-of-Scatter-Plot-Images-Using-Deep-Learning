# Classification-of-Scatter-Plot-Images-Using-Deep-Learning

Scatter plots can’t be directly noticed by visually impaired individuals, because they are usually in an imagenformat, and so they are not naturally readable by machines. To solve this problem, this paper proposesna system that can extract visual properties from scatter plot images using deep learning and imagenprocessing techniques. 

It is the first study that automatically classifies scatter plots in terms of two aspects: degree of correlation (strong or weak) and types of correlation (positive, negative, or neutral). In the experimental studies, alternative convolutional neural network (CNN) architectures were compared on both synthetic and real-world datasets in terms of accuracy, including Residual Networks (ResNet), Alex Networks (AlexNet), and Visual Geometry Group (VGG) Networks. The experimental results showed that the proposed system successfully (93.90%) classified scatter plot images to help visually impaired users understand the information given in the graph.

<img width="568" alt="Screen Shot 2023-03-13 at 17 36 15" src="https://user-images.githubusercontent.com/50110116/224733932-709e9d35-ef04-44e2-8cc8-29e7f13c309a.png">

## Create Dataset
The training dataset was not used readily in the project and it was created with python code. While creating the data, it was paid attention that the plots were not always flat. Scatter plot images can be loaded into the system incorrectly or incompletely, so visually impaired individuals can use this application. In the prepared dataset, all types of scatter plots (Strong-Positive, Strong-Negative, Non-Correlation, Weak-Positive, Weak-Negative). 

First, the model was trained with 3000 scatter plot images, 600 from each class. After that, the model was validated with 1000 images and finally tested with 1000 images, 200 from each class. In total, the dataset contains 5000 different scatter plot images for training, validation, and testing.

<img width="553" alt="Screen Shot 2023-03-13 at 18 04 36" src="https://user-images.githubusercontent.com/50110116/224742429-e208e0f4-6507-4461-869b-c5fd4c3c2e24.png">

## Create Website 
The created model is presented as a website using Streamlit. The website has been developed for visually impaired individuals.

<img width="716" alt="Screen Shot 2023-03-13 at 18 02 39" src="https://user-images.githubusercontent.com/50110116/224744028-4e043054-cab7-414d-88e9-6cb08a4d4d48.png">

## Results
The results of the models are as follows

<img width="332" alt="Screen Shot 2023-03-13 at 18 08 20" src="https://user-images.githubusercontent.com/50110116/224743363-40aefede-df2a-4fc5-b3cf-1c30ac3156fc.png">

## How To Run

In order to run the project, the environment is created first.
- python3 -m venv env

The created environment is activated.
- source env/bin/activate

The libraries in the requirements.txt file are loaded into the environment.
- pip install -r requirements.txt

Web_App is run and the project redirects us to the web page
- streamlit run Web_App.py


## Publication
[Article Link](https://dergipark.org.tr/en/download/article-file/1910064)

