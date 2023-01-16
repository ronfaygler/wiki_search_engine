from flask import Flask, request, jsonify
# import inverted_index_gcp
from inverted_index_gcp import *
from page_rank import *
from page_view import *
from anchor_title_search import *
from body_search import *
from itertools import count, groupby
import pandas as pd
import re
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from pathlib import Path
import math
import hashlib
import random
nltk.download('stopwords')
import itertools
from nltk.stem.porter import *
from time import time
from timeit import timeit
from google.cloud import storage
import pandas as pd
from collections import defaultdict, Counter
import re
import nltk
import numpy as np
import operator
from contextlib import closing
import json
from operator import itemgetter
import pickle
from itertools import chain
import time


def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()

class MyFlaskApp(Flask):
    def __init__(
            self,
            import_name,
            static_url_path=None,
            static_folder="static",
            static_host=None,
            host_matching=False,
            subdomain_matching=False,
            template_folder="templates",
            instance_path=None,
            instance_relative_config=False,
            root_path=None
    ):

        super().__init__(import_name, static_url_path, static_folder, static_host, host_matching, subdomain_matching,
                       template_folder, instance_path, instance_relative_config,root_path)
        self.d_title={}
        self.d_anchor = {}
        self.idx_body = InvertedIndex.read_index('/home/cherryn/body_index', 'body')
        self.idx_title = InvertedIndex.read_index('/home/cherryn/title_index', 'title')
        self.idx_anchor = InvertedIndex.read_index('/home/cherryn/anchor_index', 'anchor')

    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query1 = request.args.get('query', '')
    if len(query1) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    query = [token.group() for token in RE_WORD.finditer(query1.lower())]
    query = [term for term in query if term in app.idx_title.df]
    title_scores = (help_search_title(query))[:10]
    for doc_id, score in title_scores:
        if doc_id not in app.idx_title.titleDict:
            continue
        else:
            res.append((doc_id, app.idx_title.titleDict[doc_id]))
    # END SOLUTION
    return jsonify(res)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    list_of_tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    id_score_list = sim(list_of_tokens, app.idx_body, 100)
    for val in id_score_list:
        if val[0] not in app.idx_title.titleDict:
            continue
        else:
            res.append((val[0], app.idx_body.titleDict[val[0]]))
    # END SOLUTION
    return jsonify(res)


def help_search_title(query):
    id_score_dict_title = get_candidate_documents_binary(query, app.idx_title, '/home/cherryn/postings_gcp_title/postings_gcp')
    sorted_list_title = sort_binary(id_score_dict_title)
    return sorted_list_title

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords. For example, a document
        with a title that matches two distinct query words will be ranked before a
        document with a title that matches only one distinct query word,
        regardless of the number of times the term appeared in the title (or
        query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    list_of_tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    sorted_list_title = help_search_title(list_of_tokens)
    for doc_id, score in sorted_list_title:
        if doc_id not in app.idx_title.titleDict:
            continue
        else:
            res.append((doc_id, app.idx_title.titleDict[doc_id]))
    # END SOLUTION
    return jsonify(res)


def help_search_anchor(query):
    id_score_dict_anchor = get_candidate_documents_binary(query, app.idx_anchor, '/home/cherryn/postings_gcp_anchor/postings_gcp')
    sorted_list_anchor = sort_binary(id_score_dict_anchor)
    return sorted_list_anchor


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with a anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    list_of_tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    sorted_list_anchor = help_search_anchor(list_of_tokens)
    for doc_id, score in sorted_list_anchor:
        if doc_id not in app.idx_title.titleDict:
            continue
        else:
            res.append((doc_id, app.idx_title.titleDict[doc_id]))
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = search_pagerank(wiki_ids)
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = search_pageviews(wiki_ids)
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
