# Realtime ASL Translator

This is work in progress and needs lots more fine tuning. The idea is to use a webcam to capture images of ASL signs and translate them into text. The current implementation is a simple demo that uses a pre-trained model from our notebooks in the above directory. The model is a simple CNN that was trained on the ASL alphabet dataset from kaggle. The model is loaded and used to predict the sign from the webcam feed. Whilst the model is good, I need to add hand detection and tracking and better image processing to send the correct image to the model.


This demo needs opencv to be installed you can do this with conda as follows

```
conda install -c conda-forge opencv
````
