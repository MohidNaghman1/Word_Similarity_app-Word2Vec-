✅ 1. Download GloVe Embeddings
You can download GloVe (Global Vectors for Word Representation) from the official Stanford NLP site:

🔗 https://nlp.stanford.edu/data/glove.6B.zip

⚠️ The total ZIP file is around 822 MB, so make sure you have a good internet connection.

✅ 2. Extract the ZIP File
After downloading:

Right-click on glove.6B.zip

Select “Extract All…”

You’ll get the following .txt files:

glove.6B.50d.txt

glove.6B.100d.txt

glove.6B.200d.txt

glove.6B.300d.txt ✅ (we will use this one)

✅ 3. Convert GloVe to Word2Vec Format
GloVe format is not directly compatible with Gensim, so we need to convert it using a built-in Gensim utility.

Here’s how:

python

from gensim.scripts.glove2word2vec import glove2word2vec

# Paths
glove_input_file = "glove.6B.300d.txt"
word2vec_output_file = "glove.6B.300d.word2vec.txt"

# Convert
glove2word2vec(glove_input_file, word2vec_output_file)
This will create the file glove.6B.300d.word2vec.txt in the same directory — now Gensim can load it!

✅ 4. Load the Converted File in Your App
You can now load the .word2vec.txt file like this:

python

from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format("glove.6B.300d.word2vec.txt", binary=False)
✅ 5. Use It to Find Similar Words
python

model.most_similar("cricket", topn=5)
Example output:

[('batsman', 0.74), ('match', 0.72), ('innings', 0.68), ...]
🧠 Tips for Smooth Usage
Keep the GloVe .txt files inside a models/ folder to stay organized.

Add glove.6B.300d.word2vec.txt to .gitignore if it's large and already available online.
