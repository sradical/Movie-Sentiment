
from bs4 import BeautifulSoup
import regex as re
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score

def strip_html(text):
    """
    Removes html tags
    
    Arguments:
    text -- string
    
    Returns:
    text -- without html tags
    """
    
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def preprocess(text):
    """
    Performs cleaning of text (reviews) to remove html tags, non-alphabetic characters and
    punctuations and convert to lower case. 
    
    Arguments:
    text -- string, one sentence (review) from dataframe
    
    Returns:
    text -- pre-processed/cleaned text as a string
    """
    text = strip_html(text)
    text = re.sub('\[[^]]*\]', '', text)
    text = re.sub('\'', '', text)
    text = text.lower()
    
    text = text.replace('.', ' ')
    text = text.replace(',', ' ')
    text = text.replace('"', ' ')
    text = text.replace(';', ' ')
    text = text.replace('!', ' ')
    text = text.replace('?', ' ')
    text = text.replace('(', ' ')
    text = text.replace(')', ' ')
    text = text.replace('-', ' ')
    text = text.replace('--', ' ')
    text = text.replace('?', ' ')
    text = text.replace(':', ' ')
    
    pattern=r'[0-9\s]' # Remove digits
    text = re.sub(pattern, ' ', text)
    text = [word for word in text.split() if len(word) > 2]
    
    return ' '.join(text)


def sentence_to_avg(sentence, word_to_vec_map):
    """
    Converts a sentence (string) into a list of words (strings). Extracts the vector representation of 
    each word and averages its value into a single vector encoding the meaning of the sentence.
    
    Arguments:
    sentence -- string, one training example from X
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its n-dimensional vector
    representation - where n is the dimension of the word vectors
    
    Returns:
    avg -- average vector encoding information about the sentence, numpy-array of shape (n,)
    """
    
    # word2vec fitted vocabulary
    vocab = list(word_to_vec_map.vocab.keys()) 
    
    # Split sentence into list of lower case words 
    words = [word for word in sentence if word in vocab]

    # Initialize the average word vector, should have the same shape as word vectors.
    avg = np.zeros(word_to_vec_map[list(words)[0]].shape)
    
    # Average the word vectors. Loop over the words in the list "words".
    for w in words:
        avg += word_to_vec_map[w]
    avg = avg/len(words)
    
    
    return avg

def convert_to_one_hot(Y, C):
    """
    Converts a categorical variable into one hot encoded vector
    
    Arguments:
    [Y] -- List = categories of each row of dataset. 
    C -- Number of categories    
    
    Returns:
    Y -- one hot encoded vector
    """
    
    Y = np.eye(C)[Y]
    return Y


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def predict(X, Y, W, b, word_to_vec_map, vector_dim=400):
    """
    Given X sentences (reviews) and Y labels (sentiments), predict sentiments and compute the accuracy of model over the given set.
    
    Arguments:
    X -- input data containing reviews, numpy array of shape (m, None)
    Y -- labels, containing index of the label, numpy array of shape (m, 1)
    vector_dim -- dimension of the word vector
    
    Returns:
    pred -- numpy array of shape (m, 1) with predictions
    """
    m = len(X)
    pred = np.zeros((m, 1))
    
    # word2vec fitted vocabulary
    vocab = list(word_to_vec_map.vocab.keys())
    
    for j in range(m):                       # Loop over training examples
        
        # Split jth test example (sentence) into list of lower case words
        words = [word for word in X[j] if word in vocab]
        
        # Average words' vectors
        avg = np.zeros((vector_dim,))
        for w in words:
            avg += word_to_vec_map[w]
        
        if len(words) > 0:
            avg = avg/len(words)
        else:
            avg

        # Forward propagation
        Z = np.dot(W, avg) + b
        A = softmax(Z)
        pred[j] = np.argmax(A)
    
    print("Accuracy: "  + str(round(accuracy_score(pred, Y),2)))
        
    return pred


def Embedding_model(X, Y, word_to_vec_map, n_y = 2, n_h = 400, learning_rate = 0.1, num_iterations = 1):
    """
    Model to train word vector representations in numpy.
    
    Arguments:
    X -- input data, numpy array of reviews (sentences) as strings, of shape (m, 1)
    Y -- labels (sentiments), numpy array of integers between 0 and 1, numpy-array of shape (m, 1)
    n_y -- Number of classes
    n_h -- Word Vector Dimension
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    learning_rate -- learning_rate for the stochastic gradient descent algorithm
    num_iterations -- number of iterations
    
    Returns:
    pred -- vector of predictions, numpy-array of shape (m, 1)
    W -- weight matrix of the softmax layer, of shape (n_y, n_h)
    b -- bias of the softmax layer, of shape (n_y,)
    """
    
    np.random.seed(1)

    # Define number of training examples
    m = len(Y)                         
    
    # Initialize parameters using Xavier initialization
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))
    
    # Convert Y to Y_onehot with n_y classes
    Y_oh = convert_to_one_hot(list(Y), C = n_y) 
     
    
    # Optimization loop
    for t in range(num_iterations):                       # Loop over the number of iterations
        for i in range(m):                                # Loop over the training examples
            
            # Average the word vectors of the words from the i'th training example
            avg = sentence_to_avg(X[i], word_to_vec_map)

            # Forward propagate the avg through the softmax layer
            z = np.dot(W, avg) + b
            a = softmax(z)

            # Compute cost using the i'th training label's one hot representation and "A" (the output of the softmax)
            cost = - np.sum(np.multiply(Y_oh[i], np.log(a)))
                    
            # Compute gradients 
            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape(n_y,1), avg.reshape(1, n_h))
            db = dz

            # Update parameters with Stochastic Gradient Descent
            W = W - learning_rate * dW
            b = b - learning_rate * db
            
        if t % 1 == 0:
            print("Coat: " + "  = " + str(cost))
            pred = predict(X, Y, W, b, word_to_vec_map)

    return pred, W, b

