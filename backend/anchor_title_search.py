import numpy as np


def get_candidate_documents_binary(query, index, directory):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search and
    evaluate the number of different  words that appear in the document.
    Then it will populate the dictionary 'candidates.'

    Parameters:
    -----------
    query: a list of tokens

    index:  inverted index loaded from the corresponding files.

    Returns:
    -----------
    dictionary of candidates[doc id]=score

    """
    candidates = {}
    for term in np.unique(query):
        if term in index.df:
            list_of_doc = index.read_posting_list(term, directory)
            for doc_id, freq in list_of_doc:
                if doc_id in candidates.keys():
                    candidates[doc_id] += 1
                else:
                    candidates[doc_id] = 1
    return candidates


def sort_binary(candidate_dict):
    """
    Sort and return the highest N documents according to the binary score.

    Parameters:
    -----------
    sim_dict: a dictionary of binary score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: binary score.  the number of words that appear from the query

    N: Integer (how many documents to retrieve). By default N = 3

    Returns:
    -----------
    a ranked list of pairs (doc_id, score)
    """
    return sorted([(doc_id, score) for doc_id, score in candidate_dict.items()], key=lambda x: x[1], reverse=True)

