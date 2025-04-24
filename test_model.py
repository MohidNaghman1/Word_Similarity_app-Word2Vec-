from gensim.models import Word2Vec

# Load the saved model
model = Word2Vec.load("models\expanded_model_optimized.model")

# Test similar words
word = "batsman"
if word in model.wv:
    similar_words = model.wv.most_similar(word, topn=10)
    print(f"Top words similar to '{word}':")
    for word, score in similar_words:
        print(f"{word}  -->  similarity: {round(score, 2)}")
else:
    print(f"'{word}' not found in vocabulary.")
