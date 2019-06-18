# TFBindPred
TFBindPred: A statistical learning method to predict Transcription Factor - DNA binding

The pairwise contributions to the total free-energy of binding, as calculate with MMPBSA and MMGBSA, 
were used as features for statistical learning calculations. 

We employed three different statistical learning algorithms, listed here in order of complexity: Logistic Regression (LR), 
Support Vector Machines (SVM) and Convolutional Neural Networks (CNN). 

We decided to employ these three algorithms because they are based on different theories and present a different approach in the 
processing and treatment of the data used for the classification process. 
In particular LR and SVMs return models that are particularly suitable for problems with a clear boundary between 
classes in the data space, while CNN is widely used for data in matrix format, e.g. images.
