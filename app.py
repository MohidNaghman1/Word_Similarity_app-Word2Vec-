from flask import Flask, render_template, request
from gensim.models import Word2Vec, KeyedVectors
import os

app = Flask(__name__)

# Load the custom Word2Vec model
custom_model_path = "models/expanded_model_optimized.model"
custom_model = Word2Vec.load(custom_model_path)

# Pre-trained model path (GloVe converted to Word2Vec format)
pretrained_model_path = "models/glove.6B.300d.word2vec.txt"
pretrained_model = None  # Loaded on demand

@app.route('/', methods=['GET', 'POST'])
def index():
    global pretrained_model
    result = []
    word = ""
    model_type = "custom"
    top_n = 5
    error = ""

    if request.method == 'POST':
        word = request.form['word'].lower()
        model_type = request.form['model_type']
        top_n = int(request.form.get('top_n', 5))

        try:
            # Handle the model selection based on the form input
            if model_type == "pretrained":
                if pretrained_model is None:
                    pretrained_model = KeyedVectors.load_word2vec_format(pretrained_model_path, binary=False)
                result = pretrained_model.most_similar(word, topn=top_n)
            else:
                # Use the custom model
                result = custom_model.wv.most_similar(word, topn=top_n)
        except KeyError:
            error = f"'{word}' not found in the selected model's vocabulary."

    return render_template('index.html', result=result, word=word, model_type=model_type, topn=top_n, error=error)

if __name__ == '__main__':
    app.run(debug=True)
