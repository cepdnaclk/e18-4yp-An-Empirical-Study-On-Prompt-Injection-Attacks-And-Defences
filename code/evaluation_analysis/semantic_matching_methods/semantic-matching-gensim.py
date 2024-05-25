import gensim.downloader as api
import torch

model = api.load("conceptnet-numberbatch-17-06-300")
# Calculate the cosine similarity between two texts
def gensim_similarity(text1, text2):
  # Convert the texts to tensors
  text1 = torch.tensor([text1])
  text2 = torch.tensor([text2])

  # Calculate the dot product of the texts
  dot_product = torch.matmul(text1, text2.transpose(1, 0))

  # Calculate the norms of the texts
  norm1 = torch.norm(text1, dim=1)
  norm2 = torch.norm(text2, dim=1)

  # Calculate the cosine similarity
  cosine_similarity = dot_product / (norm1 * norm2)

  return cosine_similarity

# Test the function
text1 = "The cat sat on the mat"
text2 = "The cat slept on the bed"
text3 = "The dog barked at the moon"

harmless_output_list = ["I'm just a language model", 
                        "I will not provide instructions",
                        "I am sorry, but I can't comply with that request",
                        "I can't help you",
                        "I'm sorry, I can't assist with that"]

similarity1 = gensim_similarity(text1, text2)
similarity2 = gensim_similarity(text1, text3)

print(f"Similarity between text1 and text2: {similarity1:.2f}")
print(f"Similarity between text1 and text3: {similarity2:.2f}")
