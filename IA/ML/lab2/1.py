from sklearn.naive_bayes import CategoricalNB
import numpy as np

input = [(160, "F"), (165, "F"), (155, "F"), (172, "F"), (175, "B"), (180, "B"), (177, "B"), (190, "F")]

training_data = np.array([x[0] for x in input])
training_labels = np.array([x[1] for x in input])

height_bins = np.linspace(start=170, stop=190, num=4, endpoint=False)

classified_heights = np.digitize(training_data, height_bins)
classified_heights = classified_heights.reshape(len(classified_heights), 1)

NB_model = CategoricalNB()
NB_model.fit(classified_heights, training_labels)

classified_test = np.digitize([178, 170], height_bins)
classified_test = classified_test.reshape(len(classified_test), 1)

pred = NB_model.predict(classified_test)
prob = NB_model.predict_proba(classified_test)

print(pred)
print(prob)

