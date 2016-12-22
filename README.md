# reliability-diagrams

Plots a reliability diagram for the predictions made by a machine learning classifier. Given an instance x with true label y, a machine learning classifier is "well calibrated" if the confidence of the classifier is given by: confidence = P(y* = y) where y* is the classifier's prediction of the label of x. This program bins predictions according to the predictions' confidence values and plots the average confidence of predictions in each bin against the true probability that the prediction is correct.  0.95 consistency bars are drawn according to the methods described in:

Bröcker, J. and Smith, L. (2007). Increasing the reliability of reliability diagrams. Weather and Forecasting,22(3), 651–661
