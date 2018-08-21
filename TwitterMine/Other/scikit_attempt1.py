import sklearn.datasets, sys

#categories used in classification (Change with our categories)
categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

#The example data (Change with our data) 
#from sklearn.datasets import fetch_20newsgroups # this one is the example data

# Word counter for bag of words classification (KEEP)
from sklearn.feature_extraction.text import CountVectorizer 

# TF-IDF method (frequent words like 'the' count less) <- We should prob use this too (KEEP)
from sklearn.feature_extraction.text import TfidfTransformer 

#The actual algorithm used to classify (KEEP & add logistic regression?)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
# RF to use for later
from sklearn.ensemble import RandomForestClassifier

#For the accuracy report and chart (KEEP)
from sklearn import metrics
import numpy as np

"""
Command pipeline that does everything. 
 To run an algorithm change the third parameter in the text_clf global pipeline below: 
	1. Multinomial NB(), 
	2. Logistic Regression
	3. SVMs
""" 
from sklearn.pipeline import Pipeline


text_clf_svm = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, random_state=42,
                                           max_iter=5, tol=None))])



#Another pipeline example (multinomialNB) 
text_clf_nb = Pipeline([('vect', CountVectorizer()),    # <- Bag of words counter
                     ('tfidf', TfidfTransformer()),  # <- weighs frequent words like 'the' less important to classification 
                     ('clf', MultinomialNB())])      # <- The algorithm, can be logistic regression or SVM

                 


#Another pipeline example (multinomialNB) 
text_clf_lr = Pipeline([('vect', CountVectorizer()),    # <- Bag of words counter
                     ('tfidf', TfidfTransformer()),  # <- weighs frequent words like 'the' less important to classification 
                     ('clf', LogisticRegression())])      # <- The algorithm, can be logistic regression or SVM

#buffer of 2
def accuracy(predicted, target):
	total_len = len(predicted)
	correct = 0
	for i in xrange(total_len):
		if predicted[i] == target[i] or predicted[i] == target[i] - 1 or predicted[i] == target[i]+1\
		or predicted[i] == target[i] - 2 or predicted[i] == target[i]+2:
			correct += 1

	return float(correct)/total_len

        
def main(argv):
	trainpath = 'datapath2'
	testpath = 'testpath'

	#1. load data set
	files = sklearn.datasets.load_files(trainpath)
	
	
	#2. Train using the global pipeline declared above: (bag of words, tfidf, SVM) in this case
	text_clf_svm.fit(files.data, files.target)
	text_clf_nb.fit(files.data, files.target)
	text_clf_lr.fit(files.data, files.target)  

	#3. Declare the test set and test
	files_test = sklearn.datasets.load_files(testpath)
	docs_test = files_test.data
	
	print "With 5 political division subgroups:  \n"

	#4. Print Results:
	svm_predict = text_clf_svm.predict(docs_test)
	print "a) SVM: "
	print "--------"
	print "Predicted: "
	print svm_predict
	print "Actual: "
	print files_test.target
	print("Accuracy: (buffer of two)")
	print accuracy(svm_predict, files_test.target)
	print '\n'
#	print(np.mean(predicted == files_test.target))

	#print("Accuracy: (no buffer)")
	#print(np.mean(predicted == files_test.target)) #Use numpy to calculate average accuracy

	nb_predict = text_clf_nb.predict(docs_test)
	print "b) NB: "
	print "------"
	print "Predicted: "
	print nb_predict
	print "Actual: "
	print files_test.target
	print("Accuracy: (buffer of two)")
	print accuracy(nb_predict, files_test.target)
	print '\n'


	lr_predict = text_clf_lr.predict(docs_test)
	print "c) LR: "
	print "-------"
	print "Predicted: "
	print lr_predict
	print "Actual: "
	print files_test.target
	print("Accuracy: (buffer of two)")
	print accuracy(lr_predict, files_test.target)
	print '\n'


	#print("Accuracy: ")
	#print(np.mean(predicted == files_test.target)) #Use numpy to calculate average accuracy

	#5. Print awesome score report
	#print(metrics.classification_report(files_test.target, predicted, target_names=files_test.target_names))
	#print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
	

if __name__ == '__main__':
    main(sys.argv)
