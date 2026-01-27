"""
PTM Pipeline Functions with Caching

This module contains cached pipeline functions for PhraseTopicMiner to optimize
performance in the Streamlit app.
"""

import asyncio
import streamlit as st
import phrasetopicminer as ptm
from langchain_openai import ChatOpenAI
from typing import Literal, Optional, Callable, Dict, List


@st.cache_resource(show_spinner="Loading Phrase Miner...")
def load_phrase_miner(
    method: str,
    spacy_model: str,
    include_verb_phrases: bool,
    clean_markdown: bool
) -> ptm.PhraseMiner:
    """
    Load and cache a PhraseTopicMiner PhraseMiner instance.
    
    Args:
        method: POS-tagging method ('spacy' or 'nltk')
        spacy_model: spaCy model name (e.g., 'en_core_web_sm')
        include_verb_phrases: Whether to include verb phrases
        clean_markdown: Whether to clean markdown syntax
        
    Returns:
        Cached PhraseMiner instance
    """
    miner = ptm.PhraseMiner(
        method=method,
        spacy_model=spacy_model,
        include_verb_phrases=include_verb_phrases,
        clean_markdown=clean_markdown
    )
    return miner


@st.cache_resource(show_spinner="Loading Topic Modeler...")
def load_topic_modeler(
    embedding_backend: Literal["sentence_transformers", "spacy", "custom"],
    embedding_model: Optional[str] = None,
    embedding_fn: Optional[Callable] = None,
    spacy_nlp: Optional[any] = None,
    random_state: int = 42
) -> ptm.TopicModeler:
    """
    Load and cache a PhraseTopicMiner TopicModeler instance.
    
    Args:
        embedding_backend: Which embedding strategy to use
            - "sentence_transformers": Uses SentenceTransformers
            - "spacy": Uses spaCy model with .vector
            - "custom": Uses user-supplied embedding function
        embedding_model: Model name for sentence_transformers backend
        embedding_fn: Custom embedding function (List[str] -> np.ndarray)
        spacy_nlp: spaCy language object for spacy backend
        random_state: Random seed for reproducibility
        
    Returns:
        Cached TopicModeler instance
    """
    modeler = ptm.TopicModeler(
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        embedding_fn=embedding_fn,
        spacy_nlp=spacy_nlp,
        random_state=random_state
    )
    return modeler


@st.cache_data(show_spinner="Mining phrases...")  # Disabled to allow re-mining
def run_phrase_mining(
    _miner: ptm.PhraseMiner,
    docs: List[str],
    mining_config: dict  # Cache key includes this
) -> Optional[tuple]:
    """
    Run phrase mining step (Phase 1).
    
    Extracts ALL phrases without filtering - filtering happens in UI.

    # Don't actually use mining_config in the function
    # It's just for cache invalidation
    
    Args:
        _miner: PhraseMiner instance (prefixed with _ to avoid hashing)
        docs: List of document strings
        
    Returns:
        Tuple of (np_counter, vp_counters, phrase_records, sentences_by_doc) or None if mining fails
    """
    try:
        np_counter, vp_counters, phrase_records, sentences_by_doc = _miner.mine_phrases_with_types(docs)
        
        if not phrase_records:
            st.warning("No phrases were mined. Try different settings or more text.")
            return None
            
        return (np_counter, vp_counters, phrase_records, sentences_by_doc)
        
    except Exception as e:
        st.error(f"An error occurred during phrase mining: {e}")
        return None


@st.cache_data(show_spinner="Building topics...")
def run_topic_modeling(
    _modeler: ptm.TopicModeler,
    filtered_phrase_records: List,
    sentences_by_doc: Dict,
    params: Dict
) -> Optional[Dict]:
    """
    Run topic modeling step (Phase 3).
    
    Takes pre-filtered phrase_records and builds topic model.
    No phrase filtering happens here - all filtering done in UI.
    
    Args:
        _modeler: TopicModeler instance (prefixed with _ to avoid hashing)
        filtered_phrase_records: Pre-filtered list of PhraseRecord objects
        sentences_by_doc: Sentence data from mining step
        params: Dictionary of modeling parameters including:
            - pca_n_components: Number of PCA components
            - cluster_geometry: 'umap_nd' or 'umap_2d'
            - umap_n_neighbors: UMAP n_neighbors parameter
            - umap_min_dist: UMAP min_dist parameter
            - umap_cluster_n_components: UMAP target dimensions for clustering
            - clustering_algorithm: 'hdbscan' or 'kmeans'
            - hdbscan_min_cluster_size: Minimum cluster size for HDBSCAN
            - hdbscan_min_samples: Minimum samples for HDBSCAN
            - hdbscan_metric: Distance metric for HDBSCAN
            - kmeans_max_clusters: Maximum clusters for KMeans
            - viz_reducer: 'same', 'umap_2d', or 'tsne_2d'
            - tsne_perplexity: t-SNE perplexity parameter
            - tsne_learning_rate: t-SNE learning rate
            - tsne_n_iter: t-SNE iterations
            - top_n_representatives: Number of representative phrases per cluster
            - timeline_mode: 'reading_time' or 'index'
            - speech_rate_wpm: Words per minute for reading_time mode
            
    Returns:
        Dictionary containing:
            - core_result: TopicCoreResult
            - timeline_result: TopicTimelineResult
            - sentences_by_doc: Sentence data
        Returns None if modeling fails
    """
    try:
        if not filtered_phrase_records:
            st.warning("No phrases selected. Please select at least some phrases in the filtering table.")
            return None

        # Build core topic model (no phrase filtering params!)
        core_result = _modeler.fit_core(
            phrase_records=filtered_phrase_records,
            sentences_by_doc=sentences_by_doc,

            # --- phrase filtering options ---
            include_kinds=params['filter_kinds'],
            include_patterns=params['filter_patterns_np']+params['filter_patterns_vp'],
            min_freq_unigram=params['min_freq_unigram'],
            min_freq_bigram=params['min_freq_bigram'],
            min_freq_trigram_plus=params['min_freq_trigram_plus'],
            
            # Geometric pipeline
            pca_n_components=params['pca_n_components'],
            cluster_geometry=params['cluster_geometry'],
            umap_n_neighbors=params['umap_n_neighbors'],
            umap_min_dist=params['umap_min_dist'],
            umap_cluster_n_components=params['umap_cluster_n_components'],
            
            # Clustering
            clustering_algorithm=params['clustering_algorithm'],
            hdbscan_min_cluster_size=params['hdbscan_min_cluster_size'],
            hdbscan_min_samples=params['hdbscan_min_samples'],
            hdbscan_metric=params['hdbscan_metric'],
            kmeans_max_clusters=params['kmeans_max_clusters'],
            
            # Visualization
            viz_reducer=params['viz_reducer'],
            tsne_perplexity=params['tsne_perplexity'],
            tsne_learning_rate=params['tsne_learning_rate'],
            tsne_n_iter=params['tsne_n_iter'],
            
            # Representatives
            top_n_representatives=params['top_n_representatives'],
            
            verbose=False
        )
        
        if core_result.phrases_df.empty:
            st.warning("Modeling resulted in no topics. Try relaxing your filters or adjusting clustering parameters.")
            return None

        # Build timeline
        timeline_builder = ptm.TopicTimelineBuilder(
            timeline_mode=params['timeline_mode'],
            speech_rate_wpm=params['speech_rate_wpm']
        )
        timeline_result = timeline_builder.build(core_result, sentences_by_doc)

        return {
            "core_result": core_result,
            "timeline_result": timeline_result,
            "sentences_by_doc": sentences_by_doc
        }
        
    except Exception as e:
        st.error(f"An error occurred during topic modeling: {e}")
        return None


def run_llm_labeling(
    _core_result: ptm.TopicCoreResult,
    _sentences_by_doc: Dict,
    api_key: str,
    llm_model: str
) -> Optional[ptm.TopicLabelingResult]:
    """
    Generate LLM-based topic labels using OpenAI.
    
    Note: This function is NOT cached to allow different API keys/models.
    
    Args:
        _core_result: TopicCoreResult from the pipeline
        _sentences_by_doc: Sentence data
        api_key: OpenAI API key
        llm_model: OpenAI model name (e.g., 'gpt-4o-mini')
        
    Returns:
        TopicLabelingResult or None if labeling fails
    """
    try:
        lc_llm = ChatOpenAI(model=llm_model, temperature=0, api_key=api_key)
        labeler = ptm.TopicLabeler(llm=lc_llm)
        return asyncio.run(labeler.label_topics_async(_core_result, _sentences_by_doc))
    except Exception as e:
        st.error(f"Failed to generate LLM labels. Check API key/model access. Error: {e}")
        return None