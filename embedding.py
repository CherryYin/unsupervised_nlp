# # File name: Textranker
# Author: Yin Juan.
# Created: 2021-12-01
# Word or sentence embedding, word interaction and cosine similarity by fasttext.
import fasttext
import logging
import os
import numpy as np
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WordEmbedding(object):
    def __init__(self, embedding_path):
        if embedding_path == None:
            logger.error("Embedding path is neccessary, please input it.")
        self.model = fasttext.load_model(embedding_path)
        self._embedding_path = self._embedding_path
        logging.info('Done loading sent2vec model')

    def get_word_vector(self, word):
        return self.model.get_word_vector(word)

    def get_word_id(self, word):
        return self.model.get_word_id(word)
    
    def get_sentence_vectors(self, s):
        # parameter:
        #   s: sentence, splited by space
        # return: vector of the sentence
        return self.model.get_sentence_vector(s)

    def word_cos_sim(self, word1, word2):
        vector1 = np.mat(self.model.get_word_vector(word1))
        vector2 = np.mat(self.model.get_word_vector(word2))
        num = float(vector1 * vector2.T)
        denom = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        return sim
    
    def pair_cos_sim(self, pair1, pair2):
        vector1 = np.mat(self.model.get_word_vector(pair1[0]))-\
                  np.mat(self.model.get_word_vector(pair1[1]))
        vector2 = np.mat(self.model.get_word_vector(pair2[0]))-\
                  np.mat(self.model.get_word_vector(pair2[1]))

        num = float(vector1 * vector2.T)
        denom = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        return sim