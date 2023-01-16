from flask import Flask, request, jsonify
import pickle

#reading the file with the page views
with open("/home/cherryn/pageviews-202108-user.pkl", 'rb') as f:
    pageviewDict = dict(pickle.loads(f.read()))


def get_dict_pv():
  return dict(pageviewDict)


#answering wiki_ids
def search_pageviews(wiki_ids):
    pagerank_dict = get_dict_pv()
    return [pagerank_dict[elem] for elem in wiki_ids]

