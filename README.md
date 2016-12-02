# reliability-diagrams

Plots a reliability diagram using Seaborn for the results from a machine learning classification experiment. Given an instance x with true label y, a machine learning classifier is "well calibrated" if the confidence of the classifier is given by: confidence = P(y = x). This program bins predictions according to the predictions' confidence values and plots the average confidence of predictions in each bin against the true probability that the prediction is correct.  0.95 confidence intervals are drawn according to the methods described in:

Bröcker, J. and Smith, L. (2007). Increasing the reliability of reliability diagrams. Weather and Forecasting,22(3), 651–661
