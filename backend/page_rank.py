import pandas as pd
from flask import Flask, request, jsonify

colnames=['doc_id', 'prank']
pr = pd.read_csv('/home/cherryn/part-00000-173ddbd8-0527-40a6-909f-9faf9f8e8e2a-c000.csv.gz', names=colnames, compression='gzip')


def get_dict():
    return dict(zip(pr.doc_id, pr.prank))


# answer query
def search_pagerank(wikiIds):
    pagerank_dict = get_dict()
    return [pagerank_dict[elem] for elem in wikiIds if elem in pagerank_dict]

