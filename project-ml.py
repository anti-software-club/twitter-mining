import numpy as np
X = np.random.randint(5, size=(6, 100))
y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X, y)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
print(clf.predict(X[2:3]))