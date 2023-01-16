import math
from collections import Counter
import numpy as np

def sim(q,index, N=100):
  """
  Sort and return the highest N documents according to the tfidf and cosine similarity score.

  Parameters:
  -----------
  q: list of tokens of the query

  index: inverted index loaded from the corresponding files.

  N: Integer (how many documents to retrieve).

  Returns:
  -----------
  a ranked list of pairs (doc_id, score)
  """
  sim_dict={}
  lst = []
  counter_q = Counter(q)
  dict_doc={}
  for term in np.unique(q):
    if term not in index.df:
      continue
    q_tfidf = (counter_q[term]/len(q))
    pls = index.read_posting_list(term, '/home/cherryn/postings_gcp')
    for doc_id, freq in pls:
      dict_doc[doc_id] = 1
      d_tfidf=(freq/index.DL[doc_id])*math.log(len(index.DL)/index.df[term],10)
      if doc_id in sim_dict:
        sim_dict[doc_id] += q_tfidf * d_tfidf
      else:
        sim_dict[doc_id] = q_tfidf * d_tfidf
  for doc_id in dict_doc.keys():
    lst.append((doc_id, sim_dict[doc_id]*(1/len(q))*(1/index.DL[doc_id])))
  return sorted(lst, key = lambda x: x[1],reverse=True)[:N]
