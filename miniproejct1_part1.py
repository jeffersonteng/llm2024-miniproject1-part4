from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datasets import load_dataset
import time

class TextSimilarityModel:
    def __init__(self, corpus_name, rel_name, model_name='all-MiniLM-L6-v2', top_k=10):
        """
        Initialize the model with datasets and pre-trained sentence transformer.
        """
        self.model = SentenceTransformer(model_name)
        self.corpus_name = corpus_name
        self.rel_name = rel_name
        self.top_k = top_k
        self.load_data()


    def load_data(self):
        """
        Load and filter datasets based on test queries and documents.
        """
        # Load query and document datasets
        dataset_queries = load_dataset(self.corpus_name, "queries")
        dataset_docs = load_dataset(self.corpus_name, "corpus")

        # Extract queries and documents
        self.all_queries = dataset_queries["queries"]["text"]
        self.all_query_ids = dataset_queries["queries"]["_id"]
        self.all_documents = dataset_docs["corpus"]["text"]
        self.all_document_ids = dataset_docs["corpus"]["_id"]

        # Filter queries and documents based on test set
        qrels = load_dataset(self.rel_name)["test"]
        self.filtered_query_ids = set(qrels["query-id"])
        self.filtered_doc_ids = set(qrels["corpus-id"])

        self.test_queries = [q for qid, q in zip(self.all_query_ids, self.all_queries) if qid in self.filtered_query_ids]
        self.test_query_ids = [qid for qid in self.all_query_ids if qid in self.filtered_query_ids]
        self.test_documents = [doc for did, doc in zip(self.all_document_ids, self.all_documents) if did in self.filtered_doc_ids]
        self.test_document_ids = [did for did in self.all_document_ids if did in self.filtered_doc_ids]

        self.query_id_to_relevant_doc_ids = {qid: [] for qid in self.filtered_query_ids}
        for qid, doc_id in zip(qrels["query-id"], qrels["corpus-id"]):
            if qid in self.query_id_to_relevant_doc_ids:
                self.query_id_to_relevant_doc_ids[qid].append(doc_id)

    def encode_with_glove(self, glove_file_path, sentences):
        """
        Encodes sentences by averaging GloVe 50d vectors of words in each sentence.
        Return a sequence of embeddings of the sentences.
        Download the glove vectors from here. 
        https://nlp.stanford.edu/data/glove.6B.zip
        """
        #TODO Put your code here. 
        ###########################################################################
        # get GloVe 50d vectors
        vectors = {}
        with open(glove_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line_split = line.split()
                word = line_split[0] # first item in each row is the word
                vectors[word] = np.asarray(line_split[1:], dtype='float') # everything after first item is encodings
        
        # print(vectors)
        # embeddings for each sentence
        embeddings_per_sentence = []

        for sentence in sentences:
            words = sentence.split()
            word_vector = np.mean([vectors.get(word, np.zeros_like(vectors['the'])) for word in words], axis=0, dtype=np.float64)
            embeddings_per_sentence.append(word_vector)

        # print("embeddings_per_sentence", embeddings_per_sentence)
        return embeddings_per_sentence
        ###########################################################################

    def rank_documents(self, encoding_method='sentence_transformer'):
        """
        (1) Compute cosine similarity between each document and the query
        (2) Rank documents for each query and save the results in a dictionary "query_id_to_ranked_doc_ids" 
            This will be used in "mean_average_precision"
            Example format {2: [125, 673], 35: [900, 822]}
        """
        if encoding_method == 'glove':
            query_embeddings = self.encode_with_glove("glove.6B/glove.6B.300d.txt", self.test_queries)
            document_embeddings = self.encode_with_glove("glove.6B/glove.6B.300d.txt", self.test_documents)
        elif encoding_method == 'sentence_transformer':
            query_embeddings = self.model.encode(self.test_queries)
            document_embeddings = self.model.encode(self.test_documents)
        else:
            raise ValueError("Invalid encoding method. Choose 'glove' or 'sentence_transformer'.")
        
        #TODO Put your code here.
        ###########################################################################
        self.query_id_to_ranked_doc_ids = {}

        for i, query_id in enumerate(self.test_query_ids):
            query = query_embeddings[i]

            similarities = cosine_similarity([query], document_embeddings) # query_id vs all documents similarities
            # print("similarities are 0?", similarities == np.zeros_like(similarities)) # [1 x num documents elements] each value should be a decimal between 0, 1
            ranked_documents_by_index = np.argsort(similarities[0])[::-1]
            # print("ranked_documents_by_index", ranked_documents_by_index)

            ranked_doc_ids = [self.test_document_ids[document_index] for document_index in ranked_documents_by_index]
            self.query_id_to_ranked_doc_ids[query_id] = ranked_doc_ids[:self.top_k]
        ###########################################################################

    @staticmethod
    def average_precision(relevant_docs, candidate_docs):
        """
        Compute average precision for a single query.
        """
        y_true = [1 if doc_id in relevant_docs else 0 for doc_id in candidate_docs]
        precisions = [np.mean(y_true[:k+1]) for k in range(len(y_true)) if y_true[k]]
        return np.mean(precisions) if precisions else 0

    def mean_average_precision(self):
        """
        Compute mean average precision for all queries using the "average_precision" function.
        A reference: https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52
        """
         #TODO Put your code here. 
        ###########################################################################
        # got from reference
        # MAP = (sum average_precision(q)) / Q
        # q: query
        # Q: number of queries in the set
        
        map_ = 0

        for query_id, ranked_doc_ids in self.query_id_to_ranked_doc_ids.items():
            # print(query_id)
            ap = self.average_precision(self.query_id_to_relevant_doc_ids.get(query_id, []), ranked_doc_ids)
            map_ += ap        
        # print(map_)
        return map_ / len(self.query_id_to_ranked_doc_ids)
        ###########################################################################

    def show_ranking_documents(self, example_query):
        """
        (1) rank documents with given query with cosine similaritiy scores
        (2) prints the top 10 results along with its similarity score.
        
        """
        #TODO Put your code here. 
        query_embedding = self.model.encode(example_query)
        document_embeddings = self.model.encode(self.test_documents)
        ###########################################################################

        similarity = cosine_similarity([query_embedding], document_embeddings)[0]
        print(similarity.shape)

        ranked_docs = np.argsort(similarity)[::-1]
        top_10 = ranked_docs[:self.top_k]
        # print(top_10.shape)

        for doc in top_10:
            print("Document:", self.test_documents[doc], "\nSimilarity score:", similarity[doc])
        ###########################################################################


# Initialize and use the model
model = TextSimilarityModel("BeIR/nfcorpus", "BeIR/nfcorpus-qrels")

print("Ranking with sentence_transformer...")
model.rank_documents(encoding_method='sentence_transformer')
map_score = model.mean_average_precision()
print("Mean Average Precision:", map_score)

# print("Ranking with glove...")
#model.rank_documents(encoding_method='glove')
#map_score = model.mean_average_precision()
#print("Mean Average Precision:", map_score)

# model.show_ranking_documents("Breast Cancer Cells Feed on Cholesterol")