### INSTRUCTIONS ##############################################################
# To test a model:
# 1. Set the six "user hyperparameters" below.
#    a. Make sure the MODEL_NAME matches the model you want to test.
#    b. Make sure the first four hyperparameters match the model too.
#       (It's currently set up to test the "fancy" model.)
# 2. python3 test_model.py
#    a. It will automatically run inferences on my 20 test movies.
#    b. Then it will prompt you for summaries, and predict titles for them.

# The models I trained are:
#  "simple" = words, keep stops, 50dim, 32 hidden units
#  "fancy" = sentences, keep stops, 300dim, 256 hidden units
#  "title-to-title" = sentences, keep stops, 300d, 256 hidden units
#       but trained with titles as inputs instead of summaries
#       as an informal baseline of the abilities of such a model

# To train a new model:
# 1. Set the six "user hyperparameters" below.
# 2. python3 train_model.py
# 3. Wait. Keras will show the val_loss as it trains.
# 4. When Keras finishes, it will save the model locally.


### HYPERPARAMETERS ###########################################################

### THE USER MAY EDIT THESE ###
USE_SENTENCES = True        # Use sentences embeddings? (Else, word embeddings)
REMOVE_STOPS = False        # Remove stop words?
EMBEDDING_DIM = 300         # 50, 100, 200, or 300
LATENT_DIM = 256            # Size of the hidden layer.
MODEL_NAME = "fancy"        # Name of the model to train or test
EPOCHS = 32                 # Number of training epochs
### END OF USER PARAMETERS ###

# Cut off summaries at 10 sentences or 100 words
if USE_SENTENCES:
    MAX_SUMMARY_LENGTH = 10
else:
    MAX_SUMMARY_LENGTH = 100
# Cut off titles at 10 words always
MAX_TITLE_LENGTH = 10

# Training hyperparameters
BATCH_SIZE = 64

# Imports
import numpy as np
import gensim.downloader as gensim_api
import nltk

# Dictionary of stop words, for fast lookup
nltk.download('stopwords')
STOP_WORDS = {word:True for word in nltk.corpus.stopwords.words('english')}
print()

### EMBEDDINGS ################################################################

# Fetch the chosen embedding model
embedding_model = None
if EMBEDDING_DIM == 50:
    embedding_model = 'glove-wiki-gigaword-50'
elif EMBEDDING_DIM == 100:
    embedding_model = 'glove-wiki-gigaword-100'
elif EMBEDDING_DIM == 200:
    embedding_model = 'glove-wiki-gigaword-200'
elif EMBEDDING_DIM == 300:
    #model = 'word2vec-google-news-300'
    embedding_model = 'glove-wiki-gigaword-300'
else:
    print("Embedding size must be 50, 100, 200, or 300")
    exit(1)
print("Importing embedding model: " + embedding_model + "...")
wv = gensim_api.load(embedding_model)

# When you add more dimensions, equally similar words get "further apart"
# As a benchmark of word similarity...
print("'blue' and 'gray' similarity: " + str(wv.similarity('blue', 'gray')))
print()

# This "EMBEDDING SIZE" is the size of my word-vectors,
# including three dimensions for filler, start, and stop
EMBEDDING_SIZE = wv.vector_size + 3

# Filler tokens
FILLER_TOKEN = "kylefillertoken"
FILLER_EMBEDDING = np.zeros(EMBEDDING_SIZE)
FILLER_EMBEDDING[-1] = 1.0

# Start token
START_TOKEN = "kylesstarttoken"
START_EMBEDDING = np.zeros(EMBEDDING_SIZE)
START_EMBEDDING[-2] = 1.0

# Stop token
STOP_TOKEN = "kylesstoptoken"
STOP_EMBEDDING = np.zeros(EMBEDDING_SIZE)
STOP_EMBEDDING[-3] = 1.0

def text_to_word_embeddings(text: str, add_bounds: bool = False, remove_stops: bool = False) -> np.array:
    """Convert a string to an array of word embeddings.
    
    text: the string to convert
    add_bounds: whether to add start/stop tokens
    """
    # Split the text into words
    words = nltk.word_tokenize(text.lower())
    # Filter out stop words
    if remove_stops:
        words = [w for w in words if w not in STOP_WORDS]
    
    if add_bounds:
        # Add start/stop tokens
        words = [START_TOKEN] + words + [STOP_TOKEN]

    # Convert each word to an embedding
    embeddings = np.zeros((len(words), EMBEDDING_SIZE))
    for w, word in enumerate(words):
        # Special tokens
        if word == FILLER_TOKEN:
            embeddings[w, :] = FILLER_EMBEDDING
        elif word == START_TOKEN:
            embeddings[w, :] = START_EMBEDDING
        elif word == STOP_TOKEN:
            embeddings[w, :] = STOP_EMBEDDING
        else:
            # Words
            # (normalize them, so tanh can output them)
            # (note this doesn't affect cosine similarity)
            try:
                vec = wv[word]
                embeddings[w, :-3] = vec / np.linalg.norm(vec)
            except KeyError:
                # If the word isn't in the embedding model, skip it, leaving zero
                continue
    return embeddings

def word_embeddings_to_text(embeddings: np.array) -> str:
    """Convert an array of word embeddings to a string."""
    # Convert each embedding to a word
    words = []
    for embedding in embeddings:
        # Special tokens
        # The 0.1 cutoff is arbitrary but seems to work well with my models
        if embedding[-1] > 0.1:
            word = "--"
        elif embedding[-2] > 0.1:
            word = "[["
        elif embedding[-3] > 0.1:
            word = "]]"
        # Words
        else:
            #print("About to try to find the most similar word")
            word = wv.most_similar(positive=[embedding[:-3]], topn=1)[0][0]
            #print("Found it!")
        #print(embedding[-1], embedding[-2], embedding[-3], word)
        words.append(word)

    # Combine the words into a string
    text = ' '.join(words)
    return text

def text_to_sentence_embeddings(text: str, add_bounds: bool = False, remove_stops: bool = False) -> np.array:
    """Convert a string to an array of sentence embeddings.
    
    text: the string to convert
    add_bounds: whether to add start/stop tokens
    """
    # Split the text into sentences
    sentences = nltk.sent_tokenize(text.lower())
    #print("I found these sentences:")
    #print(sentences)
    
    if add_bounds:
        # Add start/stop tokens
        sentences = [START_TOKEN] + sentences + [STOP_TOKEN]

    # Convert each sentence to an embedding
    embeddings = np.zeros((len(sentences), EMBEDDING_SIZE))
    for s, sentence in enumerate(sentences):
        # Special tokens
        if sentence == START_TOKEN:
            embeddings[s, :] = START_EMBEDDING
        elif sentence == STOP_TOKEN:
            embeddings[s, :] = STOP_EMBEDDING
        else:
            # For each sentence, find the average of the word embeddings
            word_embeddings = text_to_word_embeddings(sentence, remove_stops=remove_stops)
            # If there are no embeddable words in this sentence, skip it
            # (This happens very rarely, only about 10 out of 50k movies)
            if np.all(word_embeddings == 0):
                #print("Could not embed sentence: " + sentence)
                continue
            sentence_embedding = np.mean(word_embeddings, axis=0)
            # Normalize, so tanh can output them
            sentence_embedding = sentence_embedding / np.linalg.norm(sentence_embedding)
            embeddings[s, :] = sentence_embedding
    return embeddings

### OTHER UTILITIES ###########################################################

def read_data_from_file(fname: str) -> list:
    """
    Read data from specific Kaggle file I'm using.
    
    Returns: ([titles], [summaries])"""
    f = open(fname, "r", encoding="utf8")
    titles = []
    summaries = []

    for movie in f:
        # Expecting each line to have this format:
        # ID ::: TITLE (Year) ::: GENRE ::: SYNOPSIS

        # First, pull out the "TITLE (Year)" and SYNOPSIS
        parts = movie.split(':::')
        #print(parts)
        title = parts[1].strip()
        synopsis = parts[3].strip()

        # Remove the "(Year)" from the TITLE
        # (Cut off at last open paren, then remove any space)
        if '(' in title:
            title = title[:title.rfind('(')].strip()
        
        titles.append(title)
        summaries.append(synopsis)
    return (titles, summaries)

def cosine_similarity(vec1: np.array, vec2: np.array) -> float:
    """Compute the cosine similarity between two vectors.
    No, there really isn't a convenient library with this function.
    If either is zero, return zero, like keras does.
    
    returns: the cosine similarity between the two vectors
    """
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return 0
    return vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def format_as_title(words: list) -> str:
    """Convert a list of words to a nicely-formatted Movie Title."""
    title = ""
    for w, word in enumerate(words):
        # Ignore these fakey tokens (which shouldn't be in the list anyway)
        if word in ["--", "[[", "]]"]:
            continue
        # Always capitalize the first word
        if w == 0:
            title += word.capitalize()
        # Don't capitalize these words
        elif word in ["a", "an", "the", "of", "for", "with"]:
            title += " " + word
        # No spaces before punctuation
        elif word in [".", ",", "!", "?", ":"]:
            title += word
        # No spaces before n't
        elif word == "n't":
            title += word
        # All other words are capitalized
        else:
            title += " " + word.capitalize()
    return title


### FEATURIZERS ########################################################

def featurize_summary(summary: str): # -> np.array
    """Convert a summary to a feature vector."""
    if USE_SENTENCES:
        return text_to_sentence_embeddings(summary, remove_stops=REMOVE_STOPS)
    else:
        return text_to_word_embeddings(summary, remove_stops=REMOVE_STOPS)

def featurize_title(title: str): # -> np.array
    """Convert a title to a feature vector."""
    # If the title has "quotes" around it, remove them
    # (But don't remove other quotes, eg The Boy Who Cried "Wolf!")
    if title[0] == '"' and title[-1] == '"':
        title = title[1:-1]
    return text_to_word_embeddings(title, add_bounds=True)