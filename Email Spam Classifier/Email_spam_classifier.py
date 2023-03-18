import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#Load data
DATA_frame = pd.read_csv("Emails.csv", encoding="latin-1")

#Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(DATA_frame["Emails"], DATA_frame["Classification"], test_size= 0.2)

#Vectorize the text data

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)#Fit and transform  training data
X_test = vectorizer.transform(X_test) #Transform testing data using the same vocabulary learned from training data

#Build the Naive Bayes model
naive_bayes_model = MultinomialNB() #Create a Multinomial Naive Bayes classifier object
naive_bayes_model.fit(X_train, y_train) #Train the classifier on the training data

#Evaluate the model
train_accuracy = naive_bayes_model.score(X_train, y_train) #Calculate the training accuracy
test_accuracy = naive_bayes_model.score(X_test, y_test) #Calculate the testing accuracy

print(f"training accuracy: {train_accuracy}")
print(f"testing accuracy: {test_accuracy}")

#Predictions
incoming_emails = ['Congratulations! you have won a iphone 14 pro max.',
                   'Reminder: your appointment is tomorrow.',
                   'Meeting today at 2 PM.']

incoming_emails = vectorizer.transform(incoming_emails)# Transform the incoming emails to be compatible with the vectorizer
predictions = naive_bayes_model.predict(incoming_emails)# Use the trained classifier to make predictions on the incoming emails

#Display predictions
for i in range(len(predictions)):
    print(incoming_emails[i])
    print(f"prediction: {predictions[i]}")

