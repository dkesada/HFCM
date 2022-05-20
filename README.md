# High-order Fuzzy Cognitive Maps For Multivariate Data Forecasting

## Objective

This repository is a generalization of the original fuzzy cognitive map (FCM) experimental implementation in another [repository](https://github.com/julzerinos/python-fuzzy-cognitve-maps). The idea is to make it easier to train the model and deploy it on any multivariate time series dataset. The original code is tailored to a single specific experiment and dataset, so I will make the FMC code data independent. For further information about forecasting with FMCs, [here](https://arxiv.org/abs/2201.02297v2) you can find a nice review.

## Tasks

The current roadmap of the project is:

* Create a local copy of the original repo and set it up to run the original experiment :heavy_check_mark:
* Lay down the foundation of some classes to encapsulate the whole model :heavy_check_mark:
* Simplify training: a training dataset, window size and specific parameters should return a class with a `forecast()` method :heavy_check_mark:
* Simplify testing: I have a model, I have some test data, so if I give you a starting point forecast up to horizon H :heavy_check_mark:
* Make the window move. A sliding window has less weights to train, and can make arbitrary length predictions :heavy_check_mark:

All essential task have been finished. For now, I need to test the accuracy of bigger windows and see if I can make the model work similarly to other TS forecasting models. In the end, the aim of working in HFCM was to compare them with DBNs and LSTMs.

## Deadline

Regrettably, I do not have all the time that I would like to complete this project. I need a version of this algorithm to run some tests asap, and so the level of completion of the project may vary depending on the time I have left. Eventually, I'll get around to finishing a nice, compact version of this HFCM model.

For now, the model is encapsulated inside a class that allows both training and forecasting with any dataset. It can perform arbitrary length forecasts, be saved and loaded and it can use different optimizers for training. It's mostly fine now, but I would like to delve a little bit further into the theory of HFCMs to see if the previous implementation of weight training and inference is ok or not. 
