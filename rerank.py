import h5py
import numpy as np
import time
from collections import defaultdict


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


def rerank_by_dot_product(passage_id_to_index, query_embedding, passage_embeddings, bm25_candidate_ids):
    # dot product between query and candidate embedding
    indices = []
    for passage_id in bm25_candidate_ids:
        indices.append(passage_id_to_index[passage_id])
    
    bm25_candidate_embeddings = passage_embeddings[np.array(indices)]
    scores = bm25_candidate_embeddings @ query_embedding # check candidates against the query
    order = np.argsort(scores)[::-1]

    ordered_candidates = []
    for i in order:
        ordered_candidates.append(bm25_candidate_ids[i])
    
    return ordered_candidates, scores[order]


def load_bm25_trec(trec_file):
    """
    Loads BM25 TREC run

    Returns: dict (query_id: list of passage_ids in rank order)
    """
    bm25_run = defaultdict(list)
    with open(trec_file, 'r') as f:
        for line in f:
            line = line.strip().split()
            query_id = line[0]
            doc_id = line[2]
            bm25_run[query_id].append(doc_id)
    return bm25_run


def process_bm25_and_rerank(bm25_run_file, trec_file, query_ids, query_embeddings, passage_id_to_index, passage_embeddings):
    bm25_run = load_bm25_trec(bm25_run_file)

    # rerank using embeddings
    reranked_run = defaultdict(list)
    for i, query_id in enumerate(query_ids):
        if query_id in bm25_run:
            query_embedding = query_embeddings[i]
            bm25_candidate_ids = bm25_run[query_id]

            # get embeddings and dot product scores
            ordered_candidates, scores = rerank_by_dot_product(
                passage_id_to_index,
                query_embedding,
                passage_embeddings,
                bm25_candidate_ids
            )

            for pid, score in zip(ordered_candidates, scores):
                reranked_run[query_id].append((pid, float(score)))
    
    print(f"Writing {len(reranked_run.keys())} queries to {trec_file}")
    write_trec_run(
        trec_file,
        list(reranked_run.keys()),
        [reranked_run[qid] for qid in reranked_run.keys()]
    )


def write_trec_run(filename, query_ids, ranked_lists):
    # query_ids: list of query ids in same order as ranked_lists
    # ranked_lists: list of lists of (doc_id, score)
    with open(filename, 'w') as f:
        for query_id, ranking in zip(query_ids, ranked_lists):
            for rank, (doc_id, score) in enumerate(ranking, start=1):
                f.write(f"{query_id} Q0 {doc_id} {rank} {score} BM25+Rerank\n")


def main():
    passage_filepath = 'msmarco_passages_embeddings_subset.h5'
    passage_ids, passage_embeddings = load_h5_embeddings(passage_filepath)

    query_filepath = 'msmarco_queries_dev_eval_embeddings.h5'
    query_ids, query_embeddings = load_h5_embeddings(query_filepath)
    
    passage_id_to_index = {}
    for index, passage_id in enumerate(passage_ids):
        passage_id_to_index[passage_id] = index


    # load BM25 results (top 100 per query)
    bm25_run_files = ["bm25.dev.top100.trec", "bm25.eval.one.top100.trec", "bm25.eval.two.top100.trec", "bm25.dev.top1000.trec", "bm25.eval.one.top1000.trec", "bm25.eval.two.top1000.trec"]
    trec_files = ["bm25_rerank_100.dev.trec", "bm25_rerank_100.eval.one.trec", "bm25_rerank_100.eval.two.trec", "bm25_rerank_1000.dev.trec", "bm25_rerank_1000.eval.one.trec", "bm25_rerank_1000.eval.two.trec"]
    for i in range(len(bm25_run_files)):
        run_file = bm25_run_files[i]
        trec_file = trec_files[i]
        
        start = time.time()
        
        process_bm25_and_rerank(run_file, trec_file, query_ids, query_embeddings, passage_id_to_index, passage_embeddings)
        
        elapsed = time.time() - start
        print(f"Reranked queries in {elapsed:.2f}s")


if __name__ == "__main__":
    main()