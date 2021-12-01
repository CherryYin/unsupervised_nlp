# # File name: Textranker
# Author: Yin Juan.
# Created: 2021-11-30
# Extracted keywords or key phrases by textrank algorithm.
import networkx as nx
import nltk
from pattern.en import lemma
from collections import Counter
import logging
import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Text_Rank(object):
    """ TextRank class
        Calculate relation's weight base on co-occurance
        Input yours tokens/grams which is extracted from documents kept the original order, 
        then train the process and ranking, you could get the keywords of these documentation.
        Parameters:
          words_list: The tokenized words from documents.
          vocab: vocabulary of these documents after remove words in low frequency or other invalid words.
          d: the damp of iteration
          init_value: the init score of each word
          tolerant_err: the iteration stopping limitation
          window_size: the co-occurrence window size.
          
    """
    def __init__(self, 
                 words_list, 
                 vocab,
                 d = 0.85,
                 init_value = 0.01,
                 tolerant_err = 0.0001,
                 window_size=4):
        self.vocab = set()
        
        self.G = nx.Graph()
        for words in words_list:
            words = [w for w in words if w in vocab]
            for i in range(len(words)):
                if words[i] not in self.G:
                    self.G.add_node(words[i])
                for j in range(i, min(len(words), i+window_size)):
                    if words[j] not in self.G:
                        self.G.add_node(words[j])
                    if not self.G.has_edge(words[i], words[j]):
                        self.G.add_edge(words[i], words[j], weight=1)
                    else:
                        self.G[words[i]][words[j]]['weight'] += 1
        logger.info("Word graph have been created!")
        self.d = d
        self.init_value = init_value
        self.tolerant_err = tolerant_err
        self.word_scores = {w:self.init_value for w in vocab}
                
                
    def get_neighbors(self, w):
        """Get nerighbors and weight from word-graph
           Parameters: 
             w: the word
           Return:
             neighbors: dict type
        """
        neighbors = {}
        for n in self.G.neighbors(w):
            neighbors[n] = self.G[w][n]['weight']
        return neighbors
    
    def get_neighbors_weight_sum(self, w):
        """Get w's sum weights of all nerighbors from word-graph"""
        nebr_weights = 0
        for n in self.G.neighbors(w):
            nebr_weights += self.G[w][n]['weight']
        return nebr_weights
    
    def get_all_words_neighbor_sum_weight(self):
        """Get all words' sum weights of all nerighbors"""
        self.nebr_sum_weight = {}
        for w in self.word_scores:
            self.nebr_sum_weight[w] = self.get_neighbors_weight_sum(w)

    def word_weight_org(self, w):
        """part 1 of TextRank algorithm implement"""
        weight_w = 0
        neighbors = self.get_neighbors(w)
        for n in neighbors:
            weight_n = float(neighbors[n]) / self.nebr_sum_weight[n]
            weight_n = weight_n*self.word_scores[n]
            weight_w += weight_n
        return weight_w
    
    def word_score_calc(self, w):
        """part 2 of TextRank algorithm implement.
           Add damp ratio
        """
        score = self.word_weight_org(w)
        score = score * self.d + (1-self.d)
        # self.word_scores = score
        return score
        
    def training(self):
        """ TextRank training process"""
        self.get_all_words_neighbor_sum_weight()
        while True:
            max_err = 0
            for w, ws in self.word_scores.items():
                next_ws = self.word_score_calc(w)
                if abs(ws-next_ws) > max_err:
                    max_err = abs(ws-next_ws)
                self.word_scores[w] = next_ws
            
            print(max_err)
            if max_err<=self.tolerant_err:
                break
    
    def ranking(self, top_n=20):
        """Word ranking by textrank score after training
           Parameter:
             top_n: The number of keywords or key phrases.
           Return: keywords and scores.
        """
        word_sorted = sorted(self.word_scores.items(), key=lambda x:x[1], reverse=True)
        return word_sorted[:top_n]