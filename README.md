# decision-tree

In this project, I implemented a binary decision-tree classifier entirely from scratch. The mutual information criterion was used to determine which attribute to split on at each level of the tree. 
The classifier was used to predict whether or not a patient has heart disease and also to predict the final grades of high school students. 

Files:
- inspection.py: calculates the entropy at the root of the tree and the error rate of classifying using a majority vote
- decision_tree.py: learns a decision tree with a specified maximum depth, prints the tree's structure, predicts the labels of the training and test data, and calculates errors
- education_train.tsv / education_test.tsv: data to learn and predict grades of high school students
- heart_train.tsv / heart_test.tsv: data to learn and predict patient's likelihood of heart disease 

Command Line Arguments:
python decision_tree.py < train input > < test input > < max depth > < train out > < test out > < metrics out > 

- train input: path to training input file
- test input: path to test input file
- max depth: max depth of decision tree
- train out: path to training output file where predictions on training data should be written
- test out: path to test output file where predictions on test data should be written
- metrics out: path of output file where train and test errors should be written 
