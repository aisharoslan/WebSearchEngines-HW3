import h5py
import faiss # efficient similarity search/clustering of dense vectors
import numpy as np
from collections import defaultdict

DIM = 384 # Embedding dimensions
M = 4 # max no. of neighbors per node in graph
EF_CONSTRUCTION = 50
EF_SEARCH = 50
TOPK_HNSW = 100

def load_h5_embeddings(file_path, id_key='id', embedding_key='embedding'):
    """
    Load IDs and embeddings from an HDF5 file.

    Parameters:
    - id_key: Dataset name for the IDs inside the HDF5 file.
    - embedding_key: Dataset name for the embeddings inside the HDF5 file.

    Returns:
    - ids: Numpy array of IDs (as strings).
    - embeddings: Numpy array of embeddings (as float32).
    """
    print(f"Loading data from {file_path}...")
    with h5py.File(file_path, 'r') as f:
        ids = np.array(f[id_key]).astype(str)
        embeddings = np.array(f[embedding_key]).astype(np.float32)  

    print(f"Loaded {len(ids)} embeddings.")
    return ids, embeddings

def hnsw(passage_ids, passage_embeddings, query_embeddings):
    """
    HNSW ANN search using FAISS

    Returns:
    - top_passage_ids_per_query: list of top-K passage IDs for each query
    - scores_per_query: list of corresponding similarity scores
    """
    # IndexHNSW flat - graph based ANN search
    # HNSW - best ANN for semantic retrieval
    # Supports inner product/cosine, high recall, good runtime
    # (4,50,50) to (8, 200, 200)
    
    # base vectors (passage embeddings that FAISS will search over)
    # xb = 1M x 384 = (num_passages) x (embedding_dimensions)
    xb = np.ascontiguousarray(passage_embeddings.astype("float32"))

    # build hnsw index
    # 1. init empty HNSW graph
    index = faiss.IndexHNSWFlat(DIM, M, faiss.METRIC_INNER_PRODUCT)

    # 2. how thorough graph-building is - higher recall but slower to build if higher efConstruction
    index.hnsw.efConstruction = EF_CONSTRUCTION

    # 3. insert all passage embeddings into hnsw graph, build neighborhood graph according to M and efConstruction
    index.add(xb)

    # 4. efSearch constrols how many neighbors FAISS considers during querying
    index.hnsw.efSearch = EF_SEARCH

    # query vectors (query embeddings that FAISS will search for in index)
    # xq = 1000 x 384 = (num_queries, embedding_dimensions)
    xq = np.ascontiguousarray(query_embeddings.astype("float32"))
   
    # dot_product_scores = similarity scores
    # indices = indices of top-k passages in xb for each query
    dot_product_scores, indices = index.search(xq, TOPK_HNSW)
    top_passage_ids = passage_ids[indices] # numpy will index the 2D array

    top_passage_ids_per_query = [list(row) for row in top_passage_ids]
    scores_per_query = [list(row) for row in dot_product_scores]

    return top_passage_ids_per_query, scores_per_query

    
def write_trec_run(filename, query_ids, ranked_lists):
    with open(filename, 'w') as f:
        for query_id, ranking in zip(query_ids, ranked_lists):
            for rank, (doc_id, score) in enumerate(ranking, start=1):
                f.write(f"{query_id} Q0 {doc_id} {rank} {score} HNSW\n")


def get_qrel_query_ids(qrel_path) -> set:
    query_ids = set()
    with open(qrel_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            query_id = line[0]
            query_ids.add(query_id)
    return list(query_ids)

def process_queries(qrels_file, trec_file, run_dict):
    query_ids = get_qrel_query_ids(qrels_file)

    # create ranked lists for each query
    ranked_lists = []
    valid_qrels_query_ids = []
    for query_id in query_ids:
        if query_id in run_dict:
            ranked_lists.append(run_dict[query_id])
            valid_qrels_query_ids.append(query_id)

    # write trec file
    print(f"Writing {len(valid_qrels_query_ids)} queries to {trec_file}")
    write_trec_run(
        trec_file,
        valid_qrels_query_ids,
        ranked_lists
    )

def main():
    passage_filepath = 'msmarco_passages_embeddings_subset.h5'
    passage_ids, passage_embeddings = load_h5_embeddings(passage_filepath)

    query_filepath = 'msmarco_queries_dev_eval_embeddings.h5'
    query_ids, query_embeddings = load_h5_embeddings(query_filepath)

    top_passage_ids_per_query, scores_per_query = hnsw(
        passage_ids, passage_embeddings, query_embeddings
    )

    # mapping query_id -> (ranked docs + scores)
    run_dict = {}
    for i in range(len(query_ids)):
        query_id = query_ids[i]
        passages = top_passage_ids_per_query[i]
        scores = scores_per_query[i]
        
        # sort pairs in descending order of score
        pairs = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)

        run_dict[query_id] = pairs
    
    qrels_files = ["qrels.dev.tsv", "qrels.eval.one.tsv", "qrels.eval.two.tsv"]
    trec_files = ["hnsw.low.dev.trec", "hnsw.low.eval.one.trec", "hnsw.low.eval.two.trec"]

    for i in range(len(qrels_files)):
        process_queries(qrels_files[i], trec_files[i], run_dict)


if __name__ == "__main__":
    main()