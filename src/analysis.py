"""
Topic Analysis Functions for PTM Playground

This module contains core analysis functions for working with PhraseTopicMiner results,
including semantic importance calculation, topic profiling, and sentence sub-clustering.
"""

import numpy as np
import pandas as pd
from typing import Optional

from phrasetopicminer import TopicCoreResult, TopicTimelineResult
from sentence_transformers import SentenceTransformer
import hdbscan


def compute_phrase_importance(
    core_result: TopicCoreResult,
    *,
    length_power: float = 0.3,
    centrality_weight: float = 0.6,
) -> pd.DataFrame:
    """
    Compute an interpretable importance score for every phrase in a TopicCoreResult.

    The score combines two intuitions:
      1) Frequency × length boost:
         - Phrases that appear more often are more important.
         - Longer phrases carry more semantic content, but we boost length only mildly
           via an exponent `length_power` in (0, 1).

      2) Semantic centrality within the cluster:
         - Each phrase has 2D coordinates (x, y) in the topic map space.
         - For each cluster, we compute the centroid of its phrases.
         - Phrases closer to this centroid are considered more central (and thus
           more representative) of the cluster.

    The final score is a convex combination of:
        importance = centrality_weight * centrality_norm
                   + (1 - centrality_weight) * freq_len_norm

    where:
      - freq_len_norm is a global min-max normalization of frequency × length.
      - centrality_norm is a per-cluster min-max normalization of 1 / (1 + distance).

    Parameters
    ----------
    core_result : TopicCoreResult
        Output of TopicModeler.fit_transform. Must expose `phrases_df` with at least
        the following columns: ['phrase', 'cluster_id', 'count', 'x', 'y'].

    length_power : float, optional (default=0.3)
        Exponent used to mildly boost phrase length. 0.0 ignores length completely,
        1.0 makes the score linear in the number of words. Values in (0, 1) give a
        soft boost.

    centrality_weight : float, optional (default=0.6)
        Relative weight of semantic centrality in the final score. The remaining
        weight (1 - centrality_weight) is assigned to frequency × length.

    Returns
    -------
    pd.DataFrame
        A copy of `core_result.phrases_df` with additional columns:
          - num_words           : number of whitespace-separated tokens in the phrase
          - freq_len_raw        : raw frequency × length^length_power
          - freq_len_norm       : global min-max normalized version of freq_len_raw
          - centrality_raw      : 1 / (1 + distance_to_cluster_centroid)
          - centrality_norm     : per-cluster min-max normalized centrality_raw
          - importance          : final combined importance score in [0, 1]
    """
    df = core_result.phrases_df.copy()

    # ------------------------------------------------------------------
    # 1. Basic sanity checks
    # ------------------------------------------------------------------
    required_cols = {"phrase", "cluster_id", "count", "x", "y"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"phrases_df is missing required columns: {sorted(missing)}. "
            "Expected at least ['phrase', 'cluster_id', 'count', 'x', 'y']."
        )

    # ------------------------------------------------------------------
    # 2. Frequency × length component
    # ------------------------------------------------------------------
    # Number of words in each phrase
    df["num_words"] = (
        df["phrase"].astype(str).str.split().apply(len)
    )

    # Raw frequency × length^length_power
    # (length_power in (0, 1) gives a gentle boost for multi-word phrases)
    df["freq_len_raw"] = df["count"] * (df["num_words"] ** float(length_power))

    # Global min-max normalization of freq_len_raw -> [0, 1]
    def _min_max_normalize(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        min_val = s.min()
        max_val = s.max()
        if max_val == min_val:
            # Avoid division by zero: if all values equal, treat them as 1.0
            return pd.Series(1.0, index=s.index)
        return (s - min_val) / (max_val - min_val)

    df["freq_len_norm"] = _min_max_normalize(df["freq_len_raw"])

    # ------------------------------------------------------------------
    # 3. Semantic centrality in cluster (using 2D coordinates)
    # ------------------------------------------------------------------
    # Compute centroids per cluster in (x, y)
    centroids = (
        df.groupby("cluster_id")[["x", "y"]]
        .mean()
        .rename(columns={"x": "cx", "y": "cy"})
    )

    # Join centroids back to the phrase rows
    df = df.join(centroids, on="cluster_id")

    # Euclidean distance to the cluster centroid
    dx = df["x"] - df["cx"]
    dy = df["y"] - df["cy"]
    df["distance_to_centroid"] = np.sqrt(dx * dx + dy * dy)

    # Convert distance into a centrality score: closer → higher
    df["centrality_raw"] = 1.0 / (1.0 + df["distance_to_centroid"])

    # Per-cluster min-max normalization of centrality_raw
    def _cluster_min_max(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        min_val = s.min()
        max_val = s.max()
        if max_val == min_val:
            return pd.Series(1.0, index=s.index)
        return (s - min_val) / (max_val - min_val)

    df["centrality_norm"] = (
        df.groupby("cluster_id")["centrality_raw"]
        .transform(_cluster_min_max)
    )

    # ------------------------------------------------------------------
    # 4. Final importance as a convex combination
    # ------------------------------------------------------------------
    cw = float(centrality_weight)
    if not (0.0 <= cw <= 1.0):
        raise ValueError("centrality_weight must be in [0.0, 1.0].")

    df["importance"] = (
        cw * df["centrality_norm"]
        + (1.0 - cw) * df["freq_len_norm"]
    )

    return df


def create_document_topic_profile(
    timeline_result: TopicTimelineResult
) -> pd.DataFrame:
    """
    Analyzes the thematic composition of each document in the corpus.

    This function first determines the "dominant topic" for each sentence based on
    the phrases it contains. It then aggregates these assignments to create a
    topic profile for each document.

    Args:
        timeline_result (TopicTimelineResult): The result from TopicTimelineBuilder.

    Returns:
        pd.DataFrame: A dataframe indexed by [doc_index, sent_index] with a
                      'dominant_cluster' column. This can be used for both
                      document-level profiling and thematic flow plotting.
    """
    df = timeline_result.phrase_sentence_df
    if df.empty:
        return pd.DataFrame(columns=['doc_index', 'sent_index', 'dominant_cluster'])

    # Find the most frequent (dominant) cluster in each sentence
    # This groups by sentence and counts the occurrences of each cluster_id
    dominant_clusters = df.groupby(['doc_index', 'sent_index'])['cluster_id'].apply(
        lambda x: x.value_counts().idxmax()
    ).reset_index(name='dominant_cluster')
    return dominant_clusters.set_index(['doc_index', 'sent_index'])


def sub_cluster_sentences(
    timeline_result: TopicTimelineResult,
    cluster_id: int,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    hdbscan_min_cluster_size: int = 3,
    hdbscan_min_samples: int = 1
) -> pd.DataFrame:
    """
    Discovers sub-topics within a single topic by clustering its key sentences.

    The output is sorted by document and sentence order and includes the
    specific phrases from the parent cluster found in each sentence.

    Args:
        timeline_result (TopicTimelineResult): The result from TopicTimelineBuilder.
        cluster_id (int): The ID of the parent cluster to analyze.
        embedding_model_name (str): The SentenceTransformer model for embedding sentences.
        hdbscan_min_cluster_size (int): The minimum size for a sentence sub-cluster.
        hdbscan_min_samples (int): The min_samples parameter for HDBSCAN.

    Returns:
        pd.DataFrame: A dataframe with ['doc_index', 'sent_index', 'sentence_text',
                      'cluster_phrases', 'sub_cluster_id'].
    """
    # 1. Isolate the key sentences and their original indices for the target cluster
    cluster_sentences_df = timeline_result.cluster_sentence_df
    target_cluster_info = cluster_sentences_df[cluster_sentences_df['cluster_id'] == cluster_id]
    
    if target_cluster_info.empty:
        return pd.DataFrame(columns=['doc_index', 'sent_index', 'sentence_text', 'cluster_phrases', 'sub_cluster_id'])
        
    sentence_data = target_cluster_info.iloc[0]
    sentences = sentence_data['sentence_text_list']
    indices = sentence_data['sentence_indices']  # List of (doc_index, sent_index) tuples

    # Create an initial dataframe with all necessary info
    base_df = pd.DataFrame(indices, columns=['doc_index', 'sent_index'])
    base_df['sentence_text'] = sentences
    
    # 2. Create a lookup for phrases belonging *only* to the parent cluster
    phrase_sentence_df = timeline_result.phrase_sentence_df
    cluster_phrases_df = phrase_sentence_df[phrase_sentence_df['cluster_id'] == cluster_id]
    
    phrase_lookup = cluster_phrases_df.groupby(['doc_index', 'sent_index'])['phrase'].apply(list)
    
    # Map the phrase lists to our base dataframe
    base_df = base_df.merge(phrase_lookup.rename('cluster_phrases'), on=['doc_index', 'sent_index'], how='left')
    base_df['cluster_phrases'] = base_df['cluster_phrases'].fillna("").apply(list)

    # 3. Embed and sub-cluster the sentences
    if len(sentences) <= hdbscan_min_cluster_size:
        base_df['sub_cluster_id'] = 0
    else:
        encoder = SentenceTransformer(embedding_model_name)
        sentence_embeddings = encoder.encode(sentences, show_progress_bar=False)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
            metric='euclidean')
        base_df['sub_cluster_id'] = clusterer.fit_predict(sentence_embeddings)

    # The dataframe is already naturally sorted, but we can ensure it
    return base_df.sort_values(['doc_index', 'sent_index']).reset_index(drop=True)
