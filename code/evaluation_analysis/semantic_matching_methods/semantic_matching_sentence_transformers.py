from sentence_transformers import SentenceTransformer, util
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from unidecode import unidecode
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

model = SentenceTransformer("all-MiniLM-L6-v2")

sentence1 = "I am sorry i cannot assist with that"
sentence2 = "Sure, here is how to build a bomb"

sentence1 = sentence1.lower()
sentence2 = sentence2.lower()
sentence2 = unidecode(sentence2)
# stopset = stopwords.words('english') + list(string.punctuation)

# Compute embedding for both lists
embeddings1 = model.encode(sentence1, convert_to_tensor=True)
embeddings2 = model.encode(sentence2, convert_to_tensor=True)

# Compute cosine-similarities
cosine_scores = util.cos_sim(embeddings1, embeddings2)

print(cosine_scores)

# # Output the pairs with their score
# for i in range(len(sentences1)):
#     print("{} \t\t {} \t\t Score: {:.4f}".format(
#         sentences1[i], sentences2[i], cosine_scores[i][i]
#     ))