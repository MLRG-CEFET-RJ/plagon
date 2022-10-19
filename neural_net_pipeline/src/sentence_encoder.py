from abc import ABC, abstractmethod
import numpy as np

class AbstractSentenceEncoder(ABC):

    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def get_dim(self):
        pass

    @abstractmethod
    def encode_sentence(self, sentence):
        pass
    
    @abstractmethod
    def encode_sentence_list(self, sentence_list):
        pass

class Doc2VecSentenceEncoder(AbstractSentenceEncoder):
    def __init__(self, d2v_model):
        self.d2v_model = d2v_model

    def get_dim(self):
        return 100
    
    def encode_sentence(self, sentence):
        #return self.d2v_model.infer_vector(sentence.split(), epochs=500, alpha=0.00025)
        return self.d2v_model.infer_vector(sentence.split())

    def encode_sentence_list(self, sentence_list):
        return self.d2v_model.infer_vector(sentence_list)

class UniversalSentenceEncoder(AbstractSentenceEncoder):

    def __init__(self, model):
        self.model = model

    def get_dim(self):
        return 512
 
    def _embed(self, input):
        return self.model(input)

    def encode_sentence(self, sentence):
        temp = self._embed([sentence])
        return np.array(temp[0])

    def encode_sentence_list(self, sentence_list):
        return self._embed(sentence_list)