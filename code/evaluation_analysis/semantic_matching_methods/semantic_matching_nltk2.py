import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from unidecode import unidecode
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def pre_process(text):
    sentence_h = "I am sorry, I cannot assist with that."
    # convert input corpus to lower case.
    text = text.lower()
    sentence_h = sentence_h.lower()
    text = unidecode(text)

    tokens1 = word_tokenize(text)
    tokens2 = word_tokenize(sentence_h)
    lemmatizer = WordNetLemmatizer()
    tokens1 = [lemmatizer.lemmatize(token) for token in tokens1]
    tokens2 = [lemmatizer.lemmatize(token) for token in tokens2]
    # collecting a list of stop words from nltk and punctuation form
    # string class and create single array.
    stopset = stopwords.words('english') + list(string.punctuation)
    # remove stop words and punctuations from string.
    # word_tokenize is used to tokenize the input corpus in word tokens.
    tokens1 = [token for token in tokens1 if token not in stopset]
    tokens2 = [token for token in tokens2 if token not in stopset]
    # text = " ".join([i for i in word_tokenize(text) if i not in stopset])
    # remove non-ascii characters
    
    # print(text)

    # print(words)
    # words = word_tokenize(text)
    # for w in words:
    #     print(w, " : ", lemmatizer.lemmatize(w))

    # creating vocabulary using uni-gram and bi-gram
    vectorizer = TfidfVectorizer()
    vector1 = vectorizer.fit_transform(tokens1)
    vector2 = vectorizer.transform(tokens2)

    similarity = cosine_similarity(vector1, vector2)

    return similarity

    
          
hi = pre_process("cannot answer this it does not match i am not programmed but give me any other question and I will answer i tell the truth alwas7")
print(hi)