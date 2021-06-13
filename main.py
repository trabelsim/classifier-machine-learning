import numpy as np
import re
import nltk
import sklearn
#Used for importing the dataset
from sklearn.datasets import load_files
nltk.download('stopwords')
nltk.download('wordnet')
import pickle
from nltk.corpus import stopwords
#For lemmatize the text
from nltk.stem import WordNetLemmatizer
#BagOfWords
from sklearn.feature_extraction.text import CountVectorizer
#Training and testing
from sklearn.model_selection import train_test_split
#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
#Evaluation of the model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def save_model(classifier):
    with open('model_classifier','wb') as file:
        pickle.dump(classifier, file)


def evaluate_model(category_test, category_prediction):
    print(confusion_matrix(category_test, category_prediction))
    print(classification_report(category_test,category_prediction))
    print(accuracy_score(category_test,category_prediction))


def bag_of_words(processed_data):
    #max_features - amount of features to take into consideration
    # min_df - minimum amount of texts in which the feature is appearing
    # max_df - maximum percentage of texts in which the feature is appearing
    # stop_words - words that do not contain anything useful for the processing
    vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.8, stop_words=stopwords.words('english'))
    number_representation = vectorizer.fit_transform(processed_data).toarray()
    return number_representation



#Preprocessing the texts
def text_preprocessing(unprocessed_data):
    processed_data = []
    lemmatizer = WordNetLemmatizer()
    n_loaded_files = len(unprocessed_data)
    for text_index in range(0,n_loaded_files):
        text = unprocessed_data[text_index]
        text = re.sub(r'\W',' ', str(text))
        text = re.sub(r'\s+[a-zA-Z]\s+', ' ',text)
        text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
        text = re.sub(r'\s+',' ',text,flags=re.I)
        text = re.sub(r'^b\s+', '',text)
        text = text.lower()
        text = text.split()
        text = [lemmatizer.lemmatize(word) for word in text]
        text = ' '.join(text)
        processed_data.append(text)
    return processed_data


#Loading the file from the provided path
def load_data(path):
    press_notes = load_files(path)
    press_data, press_target = press_notes.data, press_notes.target
    print(press_target)
    return press_data, press_target



loaded_data, target_categories = load_data(r"../mlarr_text")
processed_data = text_preprocessing(loaded_data)
bag_of_words_output = bag_of_words(processed_data)
bag_train, bag_test, target_train, target_test = train_test_split(bag_of_words_output, target_categories,
                                                                    test_size=0.2, random_state=0)

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(bag_train,target_train)
cat_prediction = classifier.predict(bag_test)
evaluate_model(target_test,cat_prediction)
save_model(classifier)