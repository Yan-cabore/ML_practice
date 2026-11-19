from sklearn.ensemble import RandomForestClassifier  # type: ignore

clt = RandomForestClassifier() # Model instance

X = [[1, 2, 3],
     [11, 12, 13]] # Adding 2 samples with 3 features

y = [0,1] # Classifying the samples

clt.fit(X, y) # Fitting the model

print(clt.predict(X)) # Predicts the samples of the classes utilized in the training

print(clt.predict([[11, 12, 13],[1, 2, 3]])) # Predicts new samples

