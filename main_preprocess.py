# -*- coding: utf-8 -*-
from helper_functions import Corpus
import os
import pickle
"""
    - Dataset can be downloaded here: https://github.com/marjanhs/prnn
    - Pretrained word embeddings (binary file): https://fasttext.cc/docs/en/english-vectors.html
    
"""

corpus = Corpus(T_w = 20)
corpus.extract_docs()

import numpy as np

# N_s: sentences per document
n_s_all = [len(doc) for doc in corpus.docs_L_tr + corpus.docs_R_tr]
print(f"N_s — mean: {np.mean(n_s_all):.1f}, median: {np.median(n_s_all):.1f}, "
f"p90: {np.percentile(n_s_all, 90):.0f}, p95: {np.percentile(n_s_all, 95):.0f}, max: {np.max(n_s_all)}")

# N_w: words per sentence
n_w_all = [len(sent.split()) for doc in corpus.docs_L_tr + corpus.docs_R_tr for sent in doc]
print(f"N_w — mean: {np.mean(n_w_all):.1f}, median: {np.median(n_w_all):.1f}, "
f"p90: {np.percentile(n_w_all, 90):.0f}, p95: {np.percentile(n_w_all, 95):.0f}, max: {np.max(n_w_all)}")

corpus.remove_rare_tok_chr()
corpus.make_wrd_chr_vocabularies()

with open(os.path.join("data", "data_balanced_2000_orig"), 'wb') as f:
    pickle.dump((corpus.docs_L_tr, corpus.docs_R_tr, corpus.labels_tr,
                 corpus.docs_L_te, corpus.docs_R_te, corpus.labels_te,
                 corpus.V_w, corpus.E_w, corpus.V_c), f)
