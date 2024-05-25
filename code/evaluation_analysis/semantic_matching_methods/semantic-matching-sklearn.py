import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text1 = "The cat sat on the mat"
text2 = "The cat came and sat on the mat"
text3 = "The dog barked at the moon"
text4 = "Ï am sleeping"

def sklearn_similarity(output, harmless_response):
    # Convert the texts into TF-IDF vectors
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text4])

    # Calculate the cosine similarity between the vectors
    similarity = cosine_similarity(vectors)
    return similarity

similarity_score = sklearn_similarity(text1, text2)
print(similarity_score)