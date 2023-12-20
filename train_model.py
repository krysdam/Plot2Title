import main
import numpy as np

# Import keras, but suppress the slew of warnings
print("Importing Keras...")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow import keras
print()

### PREPARE THE DATA ########################################################

# Read the movies
print("Reading movie data...")
titles, summaries = main.read_data_from_file('train_data.txt')

# While we have the data on-hand, run some statistics
#print("Average title length:", np.mean([t.count(' ') for t in titles]))
#print("Average summary length:", np.mean([t.count(' ') for t in summaries]))

#import nltk
#title_tokens = set()
#summary_tokens = set()
#for title in titles:
#    title_tokens.update(nltk.word_tokenize(title))
#for summary in summaries:
#    summary_tokens.update(nltk.word_tokenize(summary))
#print("Number of title tokens:", len(title_tokens))
#print("Number of summary tokens:", len(summary_tokens))


print("Making embeddings...")

# Actual inputs (summaries)
input_texts = [main.featurize_summary(text) for text in summaries]
#print("Average summary length: " + str(np.mean([len(t) for t in input_texts])))

# Dummy inputs that just mirror the targets (titles)
#input_texts = [utils.featurize_summary(text) for text in titles]

# Actual targets (titles)
target_texts = [main.featurize_title(text) for text in titles]
#print("Average title length: " + str(np.mean([len(t) for t in target_texts])))

# Dummy targets that are just constant
#target_texts = [utils.text_to_word_embeddings("testing one two three", add_bounds=True) for text in titles]

# Build the data for the model
print("Building encoder_input_data, decoder_input_data, and decoder_target_data...")
encoder_input_data = np.zeros(
    (len(input_texts), main.MAX_SUMMARY_LENGTH, main.EMBEDDING_SIZE), dtype="float32"
)
decoder_input_data = np.zeros(
    (len(input_texts), main.MAX_TITLE_LENGTH, main.EMBEDDING_SIZE), dtype="float32"
)
decoder_target_data = np.zeros(
    (len(input_texts), main.MAX_TITLE_LENGTH, main.EMBEDDING_SIZE), dtype="float32"
)

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    # Fill in encoder_input_data
    for w, word in enumerate(input_text):
        if w >= main.MAX_SUMMARY_LENGTH:
            break
        encoder_input_data[i, w, :] = word
    # Rest is filler
    encoder_input_data[i, w + 1 :, :] = main.FILLER_EMBEDDING

    # Fill in decoder_input_data and decoder_target_data
    # decoder_input_data is the title
    # decoder_target_data is the title, earlier by one token
    for w, word in enumerate(target_text):
        if w > main.MAX_TITLE_LENGTH:
            break
        if w < main.MAX_TITLE_LENGTH:
            decoder_input_data[i, w, :] = word
        if w > 0:
            decoder_target_data[i, w - 1, :] = word
    # Rest is filler
    decoder_input_data[i, w + 1 :, :] = main.FILLER_EMBEDDING
    decoder_target_data[i, w :, :] = main.FILLER_EMBEDDING

# Print a few examples of summary and title,
# Translated back from embeddings to text
print("Example data:")
for i in range(5):
    print("#" + str(i+1))
    print(main.word_embeddings_to_text(encoder_input_data[i]))
    print(main.word_embeddings_to_text(decoder_input_data[i]))
    print(main.word_embeddings_to_text(decoder_target_data[i]))
    print()
    print()

# While we have all the data within arm's reach, calculate a baseline.
# The baseline guess is simply the average of all targets in the train set.
# (First 80% of the data)
average_target = np.mean(decoder_target_data[ : int(len(decoder_target_data) * 0.8)], axis=0)
print("Average decoder target: " + main.word_embeddings_to_text(average_target))

# Then evaluate that baseline across the dev set (last 20% of the data).
print("Calculating baseline...")
scores = []
for target in decoder_target_data[int(len(decoder_target_data) * 0.8) : ]:
    cosine_similarities = [main.cosine_similarity(average_target[j], target[j]) for j in range(main.MAX_TITLE_LENGTH)]
    cosine_similarity = np.mean(cosine_similarities)
    scores.append(cosine_similarity)

# Average similarity is the baseline
print("Average similarity to average: " + str(np.mean(scores)))


### BUILDING THE MODEL ########################################################

# The code below this point is based on
# this keras tutorial:
# https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
# and the updated version here:
# https://keras.io/examples/nlp/lstm_seq2seq/

if __name__ == '__main__':
    # Define an input sequence and process it.
    print("Building model...")
    encoder_inputs = keras.Input(shape=(None, main.EMBEDDING_SIZE))
    encoder = keras.layers.LSTM(main.LATENT_DIM, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    # Discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = keras.Input(shape=(None, main.EMBEDDING_SIZE))

    # Set up the decoder to return full output sequences,
    # and to return internal states as well.
    decoder_lstm = keras.layers.LSTM(main.LATENT_DIM, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = keras.layers.Dense(main.EMBEDDING_SIZE, activation="tanh")
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    embedding_model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    embedding_model.summary()

    ### TRAIN THE MODEL ###
    # Only cosine_similarity makes sense, because that's what embedding similarity means.
    # Other loss functions work comparably well, though.
    # Don't report accuracy, because it's not clear what that means for embeddings.
    embedding_model.compile(
        optimizer="adam", loss="cosine_similarity", metrics=[]
    )

    print("Training model...")
    embedding_model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=main.BATCH_SIZE,
        epochs=main.EPOCHS,
        validation_split=0.2,
    )

    # Save the model
    embedding_model.save(main.MODEL_NAME)
