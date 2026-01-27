"""
PTM Playground - Source Package

This package contains core utilities for the PhraseTopicMiner Playground,
including document loading, analysis, visualizations, and pipeline functions.
"""

# Document loading
from .document_loader import (
    load_sample_documents,
    format_corpus_stats_for_display,
)

# Analysis functions
from .analysis import (
    compute_phrase_importance,
    create_document_topic_profile,
    sub_cluster_sentences,
)

# Visualization functions
from .visualizations import (
    plot_corpus_thematic_flow,
    create_topic_summary_grid,
    create_topic_wordcloud_grid,
    add_cluster_presence_overlay,
)

# Pipeline functions
from .pipeline import (
    load_phrase_miner,
    load_topic_modeler,
    run_phrase_mining,
    run_topic_modeling,
    run_llm_labeling,
)

# Utility functions
from .utils import (
    wrap_html,
    highlight_phrases_in_sentence,
    build_hover_html,
    get_phrase_count,
)

# # Results Helpers
from .results_helpers import (
    create_cluster_bar_chart,
    create_cluster_wordcloud,
    create_cluster_phrases_dataframe,
    get_phrase_occurrences,
    create_cluster_sentences_dataframe
)

__all__ = [
    # Document loading
    'load_sample_documents',
    'format_corpus_stats_for_display',
    # Analysis
    'compute_phrase_importance',
    'create_document_topic_profile',
    'sub_cluster_sentences',
    # Visualizations
    'plot_corpus_thematic_flow',
    'create_topic_summary_grid',
    'create_topic_wordcloud_grid',
    'add_cluster_presence_overlay',
    # Pipeline
    'load_phrase_miner',
    'load_topic_modeler',
    'run_phrase_mining',
    'run_topic_modeling',
    'run_llm_labeling',
    # Utils
    'wrap_html',
    'highlight_phrases_in_sentence',
    'build_hover_html',
    'get_phrase_count',
    # Results Helpers
    'create_cluster_bar_chart',
    'create_cluster_wordcloud',
    'create_cluster_phrases_dataframe',
    'get_phrase_occurrences',
    'create_cluster_sentences_dataframe',
]

__version__ = '0.1.0'