from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset

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

        # Buil query and document id to text mapping
        self.query_id_to_text = {query_id:query for query_id, query in zip(self.all_query_ids, self.all_queries)}
        self.document_id_to_text = {document_id:document for document_id, document in zip(self.all_document_ids, self.all_documents)}

        # Build relevant queries and documents mapping based on train set
        train_qrels = load_dataset(self.rel_name)["test"]
        self.train_query_id_to_relevant_doc_ids = {qid: [] for qid in train_qrels["query-id"]}
        for qid, doc_id in zip(train_qrels["query-id"], train_qrels["corpus-id"]):
            if qid in self.train_query_id_to_relevant_doc_ids:
                self.train_query_id_to_relevant_doc_ids[qid].append(doc_id)

        # Filter queries and documents and build relevant queries and documents mapping based on test set
        test_qrels = load_dataset(self.rel_name)["test"]
        self.filtered_test_query_ids = set(test_qrels["query-id"])
        self.filtered_test_doc_ids = set(test_qrels["corpus-id"])

        self.test_queries = [q for qid, q in zip(self.all_query_ids, self.all_queries) if qid in self.filtered_test_query_ids]
        self.test_query_ids = [qid for qid in self.all_query_ids if qid in self.filtered_test_query_ids]
        self.test_documents = [doc for did, doc in zip(self.all_document_ids, self.all_documents) if did in self.filtered_test_doc_ids]
        self.test_document_ids = [did for did in self.all_document_ids if did in self.filtered_test_doc_ids]


        self.test_query_id_to_relevant_doc_ids = {qid: [] for qid in self.test_query_ids}
        for qid, doc_id in zip(test_qrels["query-id"], test_qrels["corpus-id"]):
            if qid in self.test_query_id_to_relevant_doc_ids:
                self.test_query_id_to_relevant_doc_ids[qid].append(doc_id)

    def rank_documents(self):
        """
        Ranks documents for each query using the pre-trained model.
        (1) Compute cosine similarity between each document and the query
        (2) Rank documents for each query and save the results in a dictionary "query_id_to_ranked_doc_ids"
            This will be used in "mean_average_precision"
            Example format {2: [125, 673], 35: [900, 822]}
        """
        query_embeddings = self.model.encode(self.test_queries)
        document_embeddings = self.model.encode(self.test_documents)

        self.query_id_to_ranked_doc_ids = {}
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
        map_ = 0

        for query_id, ranked_doc_ids in self.query_id_to_ranked_doc_ids.items():
            # print(query_id)
            ap = self.average_precision(self.test_query_id_to_relevant_doc_ids.get(query_id, []), ranked_doc_ids)
            map_ += ap
        # print(map_)
        return map_ / len(self.query_id_to_ranked_doc_ids)
        ###########################################################################


    def fine_tune_model(self, batch_size=32, num_epochs=3, save_model_path="finetuned_senBERT"):
        """
        Fine-tunes the model using MultipleNegativesRankingLoss.
        (1) Load training data
        (2) Prepare training examples
        (3) Define a loss function
        (4) Fine-tune the model
        """
        #TODO Put your code here.
        ###########################################################################

        ###########################################################################


    def prepare_training_examples(self, train_mapping_ids):
        """
        Prepares training examples from the training data.
        """
        train_examples = []
        used_doc_ids = []
        for qid, doc_ids in self.train_query_id_to_relevant_doc_ids.items():
            usable_doc_ids = set(doc_ids) - set(used_doc_ids)
            for doc_id in usable_doc_ids:
                anchor = self.query_id_to_text[qid]
                positive = self.document_id_to_text[doc_id]
                train_examples.append(InputExample(texts=[anchor, positive]))
            used_doc_ids.extend(doc_ids)

        return train_examples


# Initialize the model

#model = TextSimilarityModel("BeIR/nfcorpus", "BeIR/nfcorpus-qrels", model_name='finetuned_senBERT_train')
model = TextSimilarityModel("BeIR/nfcorpus", "BeIR/nfcorpus-qrels", model_name='all-MiniLM-L6-v2')
print("create model")
# Finetune the model
#model.fine_tune_model(batch_size=32, num_epochs=10, save_model_path="finetuned_senBERT_train_v2")  # Adjust batch size and epochs as needed

model.rank_documents()
map_score = model.mean_average_precision()
print("Mean Average Precision:", map_score)