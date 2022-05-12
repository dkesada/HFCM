# High-order Fuzzy Cognitive Maps For Multivariate Data Forecasting

## Objective

This repository is a generalization of the original fuzzy cognitive map (FCM) experimental implementation in another [repository](https://github.com/julzerinos/python-fuzzy-cognitve-maps). The idea is to make it easier to train the model and deploy it on any multivariate time series dataset. The original code is tailored to a single specific experiment and dataset, so I will make the FMC code data independent. For further information about forecasting with FMCs, [here](https://arxiv.org/abs/2201.02297v2) you can find a nice review.

## Tasks

The current roadmap of the project is:

* Create a local copy of the original repo and set it up to run the original experiment :heavy_check_mark:
* Lay down the fundation of some classes to encapsulate the whole model
* Simplify training: a training dataset, window size and specific parameters should return a class with a `forecast()` method
* Simplify testing: I have a model, I have some test data, so if I give you a starting point forecast up to horizon H
* Make the window move. A sliding window has less weights to train, and can make arbitrary length predictions

## Deadline

Regrettably, I do not have all the time that I would like to complete this project. I need a version of this algorithm to run some tests asap, and so the level of completion of the project may vary depending on the time I have left. Eventually, I'll get around to finishing a nice, compact version of this HFCM model.

