from sklearn.neural_network import MLPClassifier
X = [[0., 0.],[0., 1.],[1., 0.], [1., 1.]]
y = [0, 1, 1, 0]

clf = MLPClassifier(solver='lbfgs', alpha=0.001, hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(X, y)

print(clf.predict_proba([[0., 1.]]))




