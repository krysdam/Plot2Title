import main
import numpy as np

# Import keras, but suppress the slew of warnings
print("Importing Keras...")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow import keras
print()

### RUN INFERENCE (DECODING) ###
# The following code is based on
# this keras tutorial:
# https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
# and the updated version here:
# https://keras.io/examples/nlp/lstm_seq2seq/

# Restore the model and construct the encoder and decoder.
model = keras.models.load_model(main.MODEL_NAME)

encoder_inputs = model.input[0]  # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = keras.Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]  # input_2
decoder_state_input_h = keras.Input(shape=(main.LATENT_DIM,), name='hinput2?')
decoder_state_input_c = keras.Input(shape=(main.LATENT_DIM,), name='cinput2?')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = keras.Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)

def decode_sequence(input_seq: np.array) -> list:
    """Give the model an input sequence, and pull out its prediction.
    
    input_seq: the input sequence, as a numpy array of embeddings
    returns: the predicted title, as a list of words
    """

    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq, verbose=0)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, main.EMBEDDING_SIZE))
    # Populate the first character of target sequence with the start token.
    target_seq[0, 0, :] = main.START_EMBEDDING

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    decoded_sentence = []
    while True:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)

        # Sample a token
        sampled_embedding = output_tokens[0, -1, :]

        # Hit max length
        if len(decoded_sentence) >= main.MAX_TITLE_LENGTH:
            break

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, main.EMBEDDING_SIZE))
        #target_seq[0, 0, :] = sampled_embedding
        # Correct the sampled embedding to the exact word:
        word = main.word_embeddings_to_text([sampled_embedding])

        decoded_sentence.append(word)
        corrected_embedding = main.text_to_word_embeddings(word, remove_stops=False)[0]
        target_seq[0, 0, :] = corrected_embedding

        # Update states
        states_value = [h, c]
    return decoded_sentence
    
# End of reliance on keras tutorial


def summary_to_title(summary: str) -> str:
    """Give the model a summary, and pull out its prediction.
    
    summary: the input summary, as a string
    returns: the predicted title, as a string
    """
    # Sequence of actual embeddings
    featurized = main.featurize_summary(summary)

    # Pad it with a couple of FILLER_EMBEDDINGs, to match the training data
    # (It performs subjectively somewhat better this way)
    input_seq = np.array([main.FILLER_EMBEDDING] * (len(featurized) + 2))
    input_seq[:len(featurized)] = featurized
    #print(utils.word_embeddings_to_text(input_seq))

    # Get the dimensions right
    input_seq = np.array([input_seq])

    # Make the title
    decoded_sentence = decode_sequence(input_seq)
    title = main.format_as_title(decoded_sentence)
    return title

### TEST ON THE DEV SET ##########################################

'''
def close_enough(s1: str, s2: str) -> bool:
    """Are the strings functionally the same?"""
    flat1 = s1.replace(" ", "").replace('"', '').lower()
    flat2 = s2.replace(" ", "").replace('"', '').lower()
    return flat1 == flat2
# Find the accuracy 

titles, summaries = utils.read_data_from_file('train_data.txt')

dev_start = int(len(titles)*0.8)
titles = titles[dev_start:]
summaries = summaries[dev_start:]
dev_size = len(titles)

correct_count = 0


for i, (title, summ) in enumerate(zip(titles, summaries)):
    if i % 100 == 0:
        print("Working on movie {} of {}".format(i, dev_size))
    #print("Working on: " + title)
    prediction = summary_to_title(summ)
    #print("Predicted title: " + prediction)

    if close_enough(title, prediction):
        correct_count += 1
        print("*" * 30)
        print("Real title:", title)
        print("Prediction:", prediction)
        print()

print("Correct: {} of {} ({}%)".format(correct_count, dev_size, correct_count/dev_size*100))
'''


### TEST ON THE TEST SET ##########################################
titles, summaries = main.read_data_from_file('test_data.txt')

for title, summ in zip(titles, summaries):
    print("Title: " + title)
    print("Summary: " + summ)
    print("Predicted title: " + summary_to_title(summ))
    print()


### TEST LIVE ###################################################
# Let the user input summaries, and give the predicted titles
while True:
    summary = input("Enter a summary: ")
    title = summary_to_title(summary)
    print("Predicted title:", title)
    print()


# Unsolved mysteries
# - Why does using [0..0] as a filler mess it up so much?
# - How did stop words make a difference before, with latent dimension of 1?
# - Why did stop words seem to matter before, at all?

# Solved mysteries
# - 54k movies * (0.8 train) / (64 per batch) = 678 batches per epoch
# - seemingly, embedding-to-embedding is typical
