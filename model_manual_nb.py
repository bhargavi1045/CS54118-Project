import math
from collections import defaultdict

class ManualNB:
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.vocab = set()
        self.word_counts = {0: defaultdict(int), 1: defaultdict(int)}
        self.class_counts = {0:0, 1:0}
        self.prior = {0:0, 1:0}
        self.V = 0

    def build_vocab(self, texts, labels):
        total_docs = len(labels)
        for text, label in zip(texts, labels):
            for word in text.split():
                self.vocab.add(word)
                self.word_counts[label][word] += 1
                self.class_counts[label] += 1
        self.V = len(self.vocab)
        self.prior[0] = sum(1 for l in labels if l==0)/total_docs
        self.prior[1] = sum(1 for l in labels if l==1)/total_docs

    def predict(self, message):
        words = message.split()
        log_prob = {}
        for c in [0,1]:
            log_prob[c] = math.log(self.prior[c])
            for word in words:
                count_w_c = self.word_counts[c].get(word,0)
                prob_w_c = (count_w_c + self.alpha)/(self.class_counts[c]+self.alpha*self.V)
                log_prob[c] += math.log(prob_w_c)
        return 1 if log_prob[1] > log_prob[0] else 0

    def predict_proba(self, message):
        words = message.split()
        log_prob = {}
        for c in [0,1]:
            log_prob[c] = math.log(self.prior[c])
            for word in words:
                count_w_c = self.word_counts[c].get(word,0)
                prob_w_c = (count_w_c + self.alpha)/(self.class_counts[c]+self.alpha*self.V)
                log_prob[c] += math.log(prob_w_c)
        max_log = max(log_prob.values())
        probs = {c: math.exp(log_prob[c]-max_log) for c in log_prob}
        s = sum(probs.values())
        return probs[1]/s
