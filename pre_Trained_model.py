from gensim.scripts.glove2word2vec import glove2word2vec

glove_input_file = 'glove.6B.300d.txt'  # Replace with the actual path
word2vec_output_file = 'models/glove.6B.300d.word2vec.txt'  # Output path

glove2word2vec(glove_input_file, word2vec_output_file)
