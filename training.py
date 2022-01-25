# *********************************GROUP NAME LIST*********************************#
# Nur Iman Kamila Binti Azharudin               1915282
# Ani Afiqah Binti Zamrud                       1912406
# Hilda Binti Mohd Fadzir                       1919268
# Nur Farisya Aqilah Binti Muhamad Fadzil       1919614

# *********************************LIBRARY*********************************#
import numpy as np
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


# Reproducibility --> to set the seed for reproducible work
np.random.seed(0xFEE1600D) #--> can put any random number so just to make sure that you are not biased. So that the code when given to someone else get the same value

# *********************************LOAD/CLEAN DATA*********************************#
# contradiction --> swapping don't with `do` and `not`, similarly with won't and `will` and `not` etc.

contractions = {
    "ain't": "am not / are not",
    "aren't": "are not / am not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had / he would",
    "he'd've": "he would have",
    "he'll": "he shall / he will",
    "he'll've": "he shall have / he will have",
    "he's": "he has / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has / how is",
    "i'd": "I had / I would",
    "i'd've": "I would have",
    "i'll": "I shall / I will",
    "i'll've": "I shall have / I will have",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it'd": "it had / it would",
    "it'd've": "it would have",
    "it'll": "it shall / it will",
    "it'll've": "it shall have / it will have",
    "it's": "it has / it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had / she would",
    "she'd've": "she would have",
    "she'll": "she shall / she will",
    "she'll've": "she shall have / she will have",
    "she's": "she has / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as / so is",
    "that'd": "that would / that had",
    "that'd've": "that would have",
    "that's": "that has / that is",
    "there'd": "there had / there would",
    "there'd've": "there would have",
    "there's": "there has / there is",
    "they'd": "they had / they would",
    "they'd've": "they would have",
    "they'll": "they shall / they will",
    "they'll've": "they shall have / they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had / we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what shall / what will",
    "what'll've": "what shall have / what will have",
    "what're": "what are",
    "what's": "what has / what is",
    "what've": "what have",
    "when's": "when has / when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has / where is",
    "where've": "where have",
    "who'll": "who shall / who will",
    "who'll've": "who shall have / who will have",
    "who's": "who has / who is",
    "who've": "who have",
    "why's": "why has / why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had / you would",
    "you'd've": "you would have",
    "you'll": "you shall / you will",
    "you'll've": "you shall have / you will have",
    "you're": "you are",
    "you've": "you have",
}

reviews = []
labels = []
with open("train.txt") as f:
    lines = f.read().splitlines()

    for line in lines:
        tokens = line.split(" ")
        label = tokens[0].replace("__label__", "")  # Remove __label__.
        text = " ".join(
            tokens[1:]
        ).lower()  # GREAT or Great -> great and join it to make it one review again
        for word in text.split():
            if word.lower() in contractions:
                text = text.replace(word, contractions[word.lower()])
        reviews.append(text)
        labels.append(label)

labels = [int(l) - 1 for l in labels]  # Change '2' and '1' to 1 and 0.
y = np.array(labels)  # x for review, y for labels

# *********************************PREPROCESSING*********************************#
# This preprocessing pipeline converts the raw data into numerical data.
# Convert data into tf-idf vectors and perform variance thresholding.
# Anything under preprocessing need to be done only once.

# ENGLISH_STOP_WORDS = minimize common words aka useless words
tfidf = TfidfVectorizer(
    stop_words=ENGLISH_STOP_WORDS  # Not including ENGLISH_STOP_WORDS
)
tfidf.fit(reviews)
x = tfidf.fit_transform(reviews)
y_train = np.array(labels)

x = x.toarray()

vt = VarianceThreshold(threshold=0.0001)
x = vt.fit_transform(x)

print(f"Data has {x.shape[0]} reviews and {x.shape[1]} features")

# *********************************TRAINING MODEL*********************************#
# Change the `test_size=` value to increase or decrease the size of the test set.

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.1)

print("\nData used for training is ", x_train.shape)
print("Data used for validating is ", x_valid.shape)

# Logistic Regression
logreg = LogisticRegression(penalty="l1", solver="saga")
logreg.fit(x_train, y_train)

# Multi Layer Perceptron
mlp = MLPClassifier()
mlp.fit(x_train, y_train)

# *********************************EVALUATE LOGREG-MODEL*********************************#
# The scores based on the True positive(TP), False positive(FP), False negative(FN), True negative(TN)

# Note: need to change the model variable here if you use logreg or mlp (For now focus on logreg)
yp_train = logreg.predict(x_train)
yp_valid = logreg.predict(x_valid)

print("\n>>This is Logistic Regression Model training set")
# Accuracy Score --> Measure for how many correct predictions your model made for the complete test dataset
print("Accuracy Score")
acc_train = accuracy_score(y_train, yp_train) * 100
acc_valid = accuracy_score(y_valid, yp_valid) * 100
print("Train accuracy", acc_train)
print("Valid accuracy", acc_valid)

# Recall Score --> Measure for how many true positives get predicted out of all the positives in the dataset
print("\nRecall Score")
rec_train = recall_score(y_train, yp_train) * 100
rec_valid = recall_score(y_valid, yp_valid) * 100
print("Train accuracy", rec_train)
print("Valid accuracy", rec_valid)

# Precision Score --> Measure for the correctness of a positive prediction/ is predicted as positive, how sure it is actually positive
print("\nPrecision Score")
pre_train = precision_score(y_train, yp_train) * 100
pre_valid = precision_score(y_valid, yp_valid) * 100
print("Train accuracy", pre_train)
print("Valid accuracy", pre_valid)

# F1 Score --> Measure a model’s accuracy based on recall and precision
print("\nF1 Score")
f1_train = f1_score(y_train, yp_train) * 100
f1_valid = f1_score(y_valid, yp_valid) * 100
print("Train F1-score", f1_train)
print("Valid F1-score", f1_valid)

# *********************************EVALUATE MLP-MODEL*********************************#
# The scores based on the True positive(TP), False positive(FP), False negative(FN), True negative(TN)

# Note: need to change the model variable here if you use logreg or mlp (For now focus on logreg)
yp_train_mlp = mlp.predict(x_train)
yp_valid_mlp = mlp.predict(x_valid)

print("\n>>This is Multi Layer Perceptron Model training set")
# Accuracy Score --> Measure for how many correct predictions your model made for the complete test dataset
print("Accuracy Score")
acc_train = accuracy_score(y_train, yp_train_mlp) * 100
acc_valid = accuracy_score(y_valid, yp_valid_mlp) * 100
print("Train accuracy", acc_train)
print("Valid accuracy", acc_valid)

# Recall Score --> Measure for how many true positives get predicted out of all the positives in the dataset
print("\nRecall Score")
rec_train = recall_score(y_train, yp_train_mlp) * 100
rec_valid = recall_score(y_valid, yp_valid_mlp) * 100
print("Train accuracy", rec_train)
print("Valid accuracy", rec_valid)

# Precision Score --> Measure for the correctness of a positive prediction/ is predicted as positive, how sure it is actually positive
print("\nPrecision Score")
pre_train = precision_score(y_train, yp_train_mlp) * 100
pre_valid = precision_score(y_valid, yp_valid_mlp) * 100
print("Train accuracy", pre_train)
print("Valid accuracy", pre_valid)

# F1 Score --> Measure a model’s accuracy based on recall and precision
print("\nF1 Score")
f1_train = f1_score(y_train, yp_train_mlp) * 100
f1_valid = f1_score(y_valid, yp_valid_mlp) * 100
print("Train F1-score", f1_train)
print("Valid F1-score", f1_valid)

# *********************************SAVE MODEL*********************************#
# Save the TfidfVectorizer and any other preprocessing too.

logreg = LogisticRegression(penalty="l1", solver="saga")
logreg.fit(x, y)
dump(logreg, "models\\logreg.joblib")
dump(mlp, "models\\mlp.joblib")
dump(tfidf, "models\\tfidf.joblib")
dump(vt, "models\\vt.joblib")
