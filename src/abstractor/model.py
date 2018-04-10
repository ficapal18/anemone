from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

from ml_component import Ml_component


class Model(Ml_component):
    
    def __init__(self):
        
        processed_data_path = '../../data/model_info/processed_compressed_data_supervised.pkl'
        self.load_data(processed_data_path)
    
    # this corresponds to our first model, the model we use to train
    def model(self, mode = 'predict', verbose = False):

        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None,), name='Raw_Encoder_Input')  # (None, max_encoder_seq_length ,1)

        x1 = Embedding(len(list(self.word2idx_en)),
                       self.embedding_dim_en,
                       weights=[self.embedding_en],
                       input_length=None,
                       trainable=False, name='First_Embedding')(encoder_inputs)  # input_length=max_encoder_seq_length
        print("shape of the encoder embedding", x1.shape)

        # encoder =LSTM(self.latent_dim,return_state=True, name= 'Encoder_LSTM')
        # encoder_outputs,state_h,state_c=encoder(x1)

        encoder = Bidirectional(LSTM(self.latent_dim, return_state=True, name='Encoder_LSTM'), merge_mode='concat')

        encoder_outputs, state_h_down, state_c_down, state_h_upper, state_c_upper = encoder(x1)
        state_c = keras.layers.concatenate([state_h_down, state_h_upper], axis=-1)
        state_h = keras.layers.concatenate([state_c_down, state_c_upper], axis=-1)

        encoder_states = [state_h, state_c]
        # ---------------------------------------------------------------------------------------------------------------#
        # Set up the decoder, using `self.encoder_states` as initial state.
        decoder_inputs = Input(shape=(None,), name='Raw_Decoder_Input')

        x2 = Embedding(len(list(self.word2idx_en)),
                       self.embedding_dim_en,
                       weights=[self.embedding_en],
                       input_length=None,
                       trainable=False, name='Second_Embedding')  # input_length=max_decoder_seq_length
        out_x2 = x2(decoder_inputs)
        # print("shape of the decoder embedding",out_x2.shape)

        decoder_lstm = LSTM(self.latent_dim * 2, return_sequences=True, return_state=True, name='Decoder_LSTM')

        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_outputs, _, _ = decoder_lstm(out_x2, initial_state=encoder_states)

        decoder_dense = Dense(len(list(self.word2idx_en)), activation='softmax', name='Dense_Layer')
        decoder_outputs = decoder_dense(decoder_outputs)
        # ---------------------------------------------------------------------------------------------------------------#
        if mode == 'train':
            # Define the model that will turn
            # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
            self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        # ---------------------------------------------------------------------------------------------------------------#
        # We have a different model to predict
        elif mode == 'predict':
            self.encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)

            decoder_state_input_h = Input(shape=(2 * self.latent_dim,))
            decoder_state_input_c = Input(shape=(2 * self.latent_dim,))
            decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

            out_x2_ = self.x2(self.decoder_inputs)

            decoder_outputs, state_h, state_c = decoder_lstm(
                out_x2, initial_state=decoder_states_inputs)  # decoder_inputs instead of x2

            decoder_states = [state_h, state_c]
            decoder_outputs = decoder_dense(decoder_outputs)

            self.decoder_model = Model(
                [decoder_inputs] + decoder_states_inputs,
                [decoder_outputs] + decoder_states)

        # ---------------------------------------------------------------------------------------------------------------#
        if verbose:
            self.model.summary()

    def load_weights(self,weights_folder = './weights_supervised', weights_file = '/weights-supervised.hdf5'):
        # load weights into new model
        self.model.load_weights(weights_folder + weights_file)
        print("Loaded model from disk")

    def train(self, weights_folder = './weights_supervised'):

        # Compile & run training
        # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.compile(optimizer=Adam(lr=self.learning_rate, decay=self.learning_rate_decay), loss='categorical_crossentropy',
                      metrics=['accuracy'])
        # ---------------------------------------------------------------------------------------------------------------#
        # Note that `decoder_target_data` needs to be one-hot encoded,
        # rather than sequences of integers like `decoder_input_data`!

        # define the checkpoint
        filepath = weights_folder + "/abstractor-weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]

        current_index = 0
        for epoch in range(0, self.epochs):
            encoder_input_data, decoder_input_data, decoder_target_data, current_index = self.get_input_and_target(self.input_texts,
                                                                                                              self.target_texts,
                                                                                                              self.max_encoder_seq_length,
                                                                                                              self.max_decoder_seq_length,
                                                                                                              self.word2idx_en,
                                                                                                              current_index,
                                                                                                              self.buffer_size)

            print("#------------------------------- EPOCH #", epoch * 3, "-----------------------------------#")
            self.model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                      batch_size=self.batch_size,
                      epochs=3,
                      validation_split=0.2, callbacks=callbacks_list)


    # this method splits the input and the targets so that they can fit in memory
    def get_input_and_target(self, input_texts, target_texts, max_encoder_seq_length, max_decoder_seq_length, word2idx_en,
                             current_index, batch_size):
        if current_index >= len(input_texts) - 1:
            current_index = 0
        if batch_size + current_index > len(input_texts) - 1:
            current_batch_length = len(input_texts) - current_index
        else:
            current_batch_length = batch_size - 1

        encoder_input_data = np.zeros(
            (current_batch_length, max_encoder_seq_length),
            dtype='float32')
        decoder_input_data = np.zeros(
            (current_batch_length, max_decoder_seq_length),
            dtype='float32')
        decoder_target_data = np.zeros(
            (current_batch_length, max_decoder_seq_length, len(list(word2idx_en))),
            dtype='float32')

        current_loop_index = 0
        for (input_text, target_text) in zip(input_texts[current_index:current_index + current_batch_length],
                                             target_texts[current_index:current_index + current_batch_length]):

            for t, word in enumerate(input_text):
                if word not in word2idx_en:
                    word = 'unknown'

                encoder_input_data[current_loop_index, t] = word2idx_en[word]

            for t, word in enumerate(target_text):
                if word not in word2idx_en:
                    word = 'unknown'
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[current_loop_index, t] = word2idx_en[word]

                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[current_loop_index, t - 1, word2idx_en[word]] = 1.
            current_loop_index += 1
        current_index = current_loop_index + current_index

        return encoder_input_data, decoder_input_data, decoder_target_data, current_index




