import pickle
import numpy as np

class Ml_component():


    def load_data(self, processed_data_path):

        with open(processed_data_path, 'rb') as handle:
            data_dict = pickle.load(handle)

        self.max_encoder_seq_length = data_dict['max_encoder_seq_length']
        self.max_decoder_seq_length = data_dict['max_decoder_seq_length']
        self.num_decoder_tokens = data_dict['num_decoder_tokens']
        # self.input_texts = data_dict['input_texts']
        # self.target_texts = data_dict['target_texts']
        self.max_encoder_seq_length = data_dict['max_encoder_seq_length']
        self.vocab_dim = data_dict['vocab_dim']
        self.idx2word_en = data_dict['idx2word_en']
        self.word2idx_en = data_dict['word2idx_en']
        self.embd_en = data_dict['embd_en']
        self.vocab_en = data_dict['vocab_en']
        self.batch_size = data_dict['batch_size']
        self.epochs = data_dict['epochs']
        self.num_samples = data_dict['num_samples']
        self.starting_line = data_dict['starting_line']
        self.learning_rate = data_dict['learning_rate']
        self.learning_rate_decay = data_dict['learning_rate_decay']
        self.vocab_dim = data_dict['vocab_dim']
        self.latent_dim = data_dict['latent_dim']
        self.vocab_size_en = len(self.vocab_en)
        self.embedding_dim_en = len(self.embd_en[0])
        self.embedding_en = np.asarray(self.embd_en)

        print("The data has been loaded from disk.")