# wiki_search_engine
# Building a search engine for English Wikipedia - IR project
Search engine for entire wikipedia corpus made as minor project in course taken in semester 5, Information Retrieval  course.

## **Utility**
- parses wikipedia dump and makes inverted index
- merges index files and split them into smaller chunks
- main query program that returns results in less than 0.1 second

## **Data**

● Entire Wikipedia dump in a shared Google Storage bucket.

● Pageviews for articles.

● Queries and a ranked list of up to 100 relevant results for them, split into train (30
queries+results given to you in new_train.json) and test (held out for evaluation).

## **code structure**
● search_frontend.py: Flask app for search engine frontend.
● inverted_index_gcp.py: indexer of documents which creates postings lists for the search engine over GCP

## **functionality**
search()- based on title search, returns up to a 10 of the best search results for the query

get_pagerank()-Returns PageRank values for a list of provided wiki article IDs

get_pageview() - Returns the number of page views that each of the provide wiki articles had

get_candidate_documents_binary(query, index, directory) & sort_binary(candidate_dict) - implementation of the title and anchor search

sim(q,index, N=100) - implementation of the body search using cosine similarity for all the wiki articles
