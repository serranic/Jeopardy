import csv

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import matplotlib.pyplot as plt

MAX_WORDS = 1500
EMBEDDING_DIM = 100
MAXLEN = 100

def plotData(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()

def buildModel(embedding_matrix, train, values):
    model = Sequential()
    model.add(Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAXLEN))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    x_train, y_train = train
    history = model.fit(x_train, y_train,
                        epochs=10, batch_size=32,
                        validation_data=values)
    model.save_weights('pre_trained_glove_model.h5')

    return (history, model)

def getGloveEmbedding(tokenizer):
    embeddings_index = {}
    f = open('/Users/nicholas/Desktop/Classes/Jeopardy Project/glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print len(embeddings_index)

    embedding_matrix = np.zeros((MAX_WORDS, EMBEDDING_DIM))
    for word,i in tokenizer.word_index.items():
        if i < MAX_WORDS:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    return embedding_matrix 

def tokenizeQuestions(question_samples, labels):
    """ Takes list of questions as input tokenizes them. pg. 183"""
    print("Initializing tokenizer")
    training_samples = 1000
    validatioin_samples = 10000
    tokenizer = Tokenizer(num_words=MAX_WORDS) # High due to obscure nouns in questions
    tokenizer.fit_on_texts(question_samples)

    sequences = tokenizer.texts_to_sequences(question_samples)
    print('Found {0} unique tokens.'.format(len(tokenizer.word_index)))
    
    data = pad_sequences(sequences, maxlen=MAXLEN)
    labels = np.asarray(labels)
    
    x_train = data[:training_samples]
    y_train = labels[:training_samples]
    x_val = data[training_samples: training_samples + validatioin_samples]
    y_val = labels[training_samples: training_samples + validatioin_samples]

    print('Shape of data tensor: {0}'.format(data.shape))
    print('Shape of label tensor: {0}'.format(labels.shape))
    
    return (tokenizer, (x_train, y_train), (x_val, y_val))

def getBagofWords(start=0, end_questions=11000): # Keep sample size small, did 1:4 split
    print("Reading questions from excel spreadsheet.")
    with open('JEOPARDY_CSV.csv', 'rb') as csvfile:
        j_reader = csv.reader(csvfile)
        # ['Show', 'Number,', 'Air', 'Date,', 'Round,', 'Category,', 'Value,', 'Question,', 'Answer']
        questions_array, labels = [], []
        for i, row in enumerate(j_reader):
            if i > start and i < end_questions:
                if 'sport' in row[3].lower() and 'transport' not in row[3].lower():
                    # Sports Questions are labeled with 1, took catgory SPORTS because
                    # that will be most reliable label
                    labels.append(1)
                else:
                    labels.append(0)
                questions_array.append(' '.join([w.lower() for w in row[5].split(' ')]))

    print("Finished reading questions from excel spreadsheet.")
    print("Sports Questions / Total Questions : {0} / {1}".format(sum(labels), len(questions_array)))
    return questions_array, labels

def testModel(tokenizer, model, test_set, test_labels):
    print("Testing Model")
    sequences = tokenizer.texts_to_sequences(test_set)
    x_test = pad_sequences(sequences, maxlen=MAXLEN)
    y_test = np.asarray(test_labels)

    model.load_weights('pre_trained_glove_model.h5')
    print(model.evaluate(x_test, y_test))

def main():
    texts, labels = getBagofWords()
    tokenizer, train, values  = tokenizeQuestions(texts, labels)
    matrix = getGloveEmbedding(tokenizer)
    history, model = buildModel(matrix, train, values)
    #  plotData(history)
    test_set, test_labels = getBagofWords(164000, 164100)
    testModel(tokenizer, model, test_set, test_labels)

if __name__ == '__main__':
    main()

