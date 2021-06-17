# Predictive Models

This folder contains all necessary code for training and implementing your own predictive models for this bot. 

# [24-hour predictor.py](https://github.com/Gary-McC/TraderbotFinal/blob/main/Predictive%20Model/24-hour%20predictor%20all%20data.py)

This file trains a new transformer based predictive model on data given to it. The model takes 24-hour ticks of Open, Close, High, Low, and Volume data and outputs the predicted Open, Close, High, and Low outputs for the next day. Because transformers are data hungry, it is best to train new models on multiple years worth of data if possible. The model ensures data is stationary by cleaning the data to include a rolling mean as well as a percent change in price rather than an absolute one. Nonetheless, it is best to use transfer learning to train an established model created with this program on the most up to date data to ensure its predictions are accurate. To run, simply ensure the the link to the input data is properly input into the program. 

# [24-hour daily predictor.py](https://github.com/Gary-McC/TraderbotFinal/blob/main/Predictive%20Model/24-hour%20daily%20predictor%20all%20data.py)

This file uses transfer learning to train an already established transformer model on the most recent asset data to ensure quality predictions in live use cases. To run, simply ensure that the Model_Name variable points to the existing model. The new model will be saved in the same folder as the old model with the date it was trained on appended to the filename.

# Transformer+TimeEmbedding+BTC-USD-24-hour-all.hdf5

Included in this folder is a pre-trained predictive model on three years of bitcoin data. Feel free to use this for your own transfer learning purposes.
