
import json
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import random


nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
intents = {
    "greeting": {
        "patterns": ["Hi", "Hello", "How are you?"],
        "responses": ["Hello!", "Hi there!", "Greetings!"]
    },
    "goodbye": {
        "patterns": ["Bye", "See you later", "Goodbye"],
        "responses": ["Goodbye!", "See you later!", "Have a great day!"]
    }
}



words = []
classes = []
documents = []
ignore_words = ['?', '!']

for intent in intents:
    for pattern in intents[intent]['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent))
        if intent not in classes:
            classes.append(intent)

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))










training_sentences = []
training_labels = []

for doc in documents:
    training_sentences.append(' '.join(doc[0]))
    training_labels.append(doc[1])

vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split())
classifier = LogisticRegression()

model = Pipeline([('vectorizer', vectorizer), ('classifier', classifier)])
model.fit(training_sentences, training_labels)







import random
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return ' '.join(sentence_words)

def predict_class(sentence, model):
    sentence = clean_up_sentence(sentence)
    return model.predict([sentence])[0]

def get_response(tag, intents):
    for intent in intents:
        if intent == tag:
            return random.choice(intents[intent]['responses'])

print("Chatbot is running!")

while True:
    message = input("")
    tag = predict_class(message, model)
    res = get_response(tag, intents)
    print(res)

