"""
PhraseTopicMiner Playground - Interactive Topic Modeling Pipeline

An educational Streamlit application for exploring phrase-centric topic modeling.
Provides full control over the PhraseTopicMiner pipeline with guided workflow.

Architecture:
    Phase 1: Data Input - Load or provide text data
    Phase 2: Phrase Mining - Extract phrases with POS patterns
    Phase 3: Phrase Filtering - Interactive review & selection
    Phase 4: Topic Modeling - Build topics from filtered phrases
    Phase 5: LLM Labeling - Generate human-friendly labels (optional)
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import tempfile
import os
import re
from typing import List, Optional
import matplotlib.pyplot as plt

import phrasetopicminer as ptm

# --- Load API Keys from Secrets ---
from dotenv import load_dotenv
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Import local modules
from src import (
    # Document loading
    load_sample_documents,
    format_corpus_stats_for_display,
    # Analysis
    compute_phrase_importance,
    create_document_topic_profile,
    sub_cluster_sentences,
    # Visualizations
    plot_corpus_thematic_flow,
    create_topic_summary_grid,
    create_topic_wordcloud_grid,
    add_cluster_presence_overlay,
    # Pipeline
    load_phrase_miner,
    load_topic_modeler,
    run_phrase_mining,
    run_topic_modeling,
    run_llm_labeling,
    # Results Helpers
    create_cluster_bar_chart,
    create_cluster_wordcloud,
    create_cluster_phrases_dataframe,
    get_phrase_occurrences,
    create_cluster_sentences_dataframe
)


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    layout="wide",
    page_title="PhraseTopicMiner Playground",
    page_icon="🔬",
    initial_sidebar_state="expanded"
)

# Clean, minimal styling
st.markdown("""
<style>
    .block-container { 
        padding-top: 1.5rem; 
        padding-bottom: 2rem; 
    }
    h1, h2, h3 { 
        font-weight: 600; 
        line-height: 1.2; 
    }
    h1 { color: #1e293b; }
    h2 { color: #334155; margin-top: 2rem; }
    h3 { color: #475569; margin-top: 1.5rem; }
    .stButton>button { 
        border-radius: 6px;
        font-weight: 500;
    }
    div[data-testid="stMetricValue"] { 
        font-size: 1.5rem;
        font-weight: 600;
    }
    /* Phase progress */
    .phase-chip {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-right: 8px;
    }
    .phase-complete {
        background: #dcfce7;
        color: #166534;
    }
    .phase-active {
        background: #dbeafe;
        color: #1e40af;
    }
    .phase-pending {
        background: #f3f4f6;
        color: #6b7280;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# UI HELPER FUNCTIONS
# ============================================================================

def render_progress_tracker():
    """Simple progress indicator showing current pipeline state."""
    phases = [
        ("Data Input", st.session_state.data_ready),
        ("Phrase Mining", st.session_state.mining_complete),
        ("Phrase Filtering", st.session_state.mining_complete),
        ("Topic Modeling", st.session_state.results is not None),
    ]
    
    st.markdown("**Pipeline Progress:**")
    
    chips = []
    for name, complete in phases:
        if complete:
            chips.append(f'<span class="phase-chip phase-complete">✓ {name}</span>')
        else:
            chips.append(f'<span class="phase-chip phase-pending">{name}</span>')
    
    st.markdown("**➙** ".join(chips), unsafe_allow_html=True)
    st.markdown("---")

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables with defaults."""
    defaults = {
        # Phase 1: Phrase Mining Results
        "phrase_records": None,
        "sentences_by_doc": None,
        "np_counter": None,
        "vp_counters": None,
        "mining_complete": False,
        
        # Phase 2: Phrase Filtering State
        "phrase_df": None,
        "filtered_phrase_records": None,
        
        # Phase 3: Topic Modeling Results
        "results": None,
        "labeling_result": None,
        "selected_cluster_id": None,
        
        # Data Input
        "data_source": "Use Example Corpus",
        "text_input": "",
        "uploaded_docs": None,
        "data_expander_open": True,
        "data_ready": False,  # tracks if data is submitted and ready
        "data_source_just_switched": False,  # Track if user just switched
        
        # PHASE 1: Mining Parameters
        "method": "spacy",
        "spacy_model": "en_core_web_lg",
        "include_verb_phrases": True,
        "clean_markdown": True,
        "last_mining_config": None,
        
        # PHASE 2: Filtering Parameters (moved from sidebar)
        "filter_kinds": ["NP", "VP"],
        "filter_patterns_np": ["BaseNP", "NP+PP", "NP+multiPP"],
        "filter_patterns_vp": ["VerbObj"],
        "min_freq_unigram": 3,
        "min_freq_bigram": 2,
        "min_freq_trigram_plus": 1,
        
        # PHASE 3: Modeling Parameters - Embedding
        "embedding_backend": "sentence_transformers",
        "embedding_model": "all-MiniLM-L6-v2",
        
        # PHASE 3: Modeling Parameters - Dimensionality Reduction
        "pca_n_components": 50,
        "cluster_geometry": "umap_nd",
        "umap_n_neighbors": 10,
        "umap_min_dist": 0.05,
        "umap_cluster_n_components": 10,
        
        # PHASE 3: Modeling Parameters - Clustering
        "clustering_algorithm": "hdbscan",
        "hdbscan_min_cluster_size": 10,
        "hdbscan_min_samples": 3,
        "hdbscan_metric": "euclidean",
        "kmeans_max_clusters": 30,
        
        # PHASE 3: Modeling Parameters - Visualization
        "viz_reducer": "umap_2d",
        "tsne_perplexity": 30.0,
        "tsne_learning_rate": 200.0,
        "tsne_n_iter": 1000,
        
        # PHASE 3: Modeling Parameters - Representatives
        "top_n_representatives": 30,
        
        # Timeline
        "timeline_mode": "reading_time",
        "speech_rate_wpm": 160,

        # PHASE 4: LLM (simplified)
        "openai_api_key_llm": "",
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_docs_from_input() -> List[str]:
    """Extract documents from user's selected input source."""
    docs = []
    source = st.session_state.data_source
    
    if source == "Use Example Corpus":
        sample_data = load_sample_documents("examples/sample_corpus")
        docs = sample_data['documents']
        
    elif source == "Paste Text":
        text = st.session_state.get('text_input', '')
        if text:
            docs = [p.strip() for p in text.split('===') if p.strip()]
            
    elif source == "Upload File(s)":
        if st.session_state.get('uploaded_docs'):
            docs = st.session_state.uploaded_docs
    
    return docs


def build_phrase_dataframe(phrase_records: List) -> pd.DataFrame:
    """
    Convert phrase_records to DataFrame for interactive editing.
    
    Args:
        phrase_records: List of PhraseRecord objects from miner
        
    Returns:
        DataFrame with columns: selected, phrase, count, doc_freq, kind, pattern, num_words
    """
    if not phrase_records:
        return pd.DataFrame()
    
    # Aggregate phrase_records by (phrase, kind, pattern)
    # Each PhraseRecord is a single occurrence, so we need to count them
    phrase_data = {}
    has_doc_id = False
    
    for rec in phrase_records:
        key = (rec.phrase, rec.kind, rec.pattern)
        
        if key not in phrase_data:
            phrase_data[key] = {
                'phrase': rec.phrase,
                'kind': rec.kind,
                'pattern': rec.pattern,
                'count': 0,
                'doc_ids': set()
            }
        
        phrase_data[key]['count'] += 1
        
        # Track unique documents - try different possible attribute names
        doc_id = None
        for attr in ['doc_id', 'doc_index', 'document_id', 'doc']:
            if hasattr(rec, attr):
                doc_id = getattr(rec, attr)
                has_doc_id = True
                break
        
        if doc_id is not None:
            phrase_data[key]['doc_ids'].add(doc_id)
    
    # Build DataFrame
    data = []
    for phrase_info in phrase_data.values():
        # If we couldn't track doc_ids, use count as a proxy
        doc_freq = len(phrase_info['doc_ids']) if has_doc_id and phrase_info['doc_ids'] else 1
        
        data.append({
            'selected': True,  # Default: all selected
            'phrase': phrase_info['phrase'],
            'count': phrase_info['count'],
            'doc_freq': doc_freq,
            'kind': phrase_info['kind'],
            'pattern': phrase_info['pattern'],
            'num_words': len(phrase_info['phrase'].split())
        })
    
    df = pd.DataFrame(data)
    
    # Sort by count descending (most frequent first)
    if not df.empty:
        df = df.sort_values('count', ascending=False).reset_index(drop=True)
    
    return df


def filter_phrase_records(
    phrase_records: List,
    phrase_df: pd.DataFrame,
    include_kinds: List[str],
    include_patterns: List[str],
    min_freq_unigram: int,
    min_freq_bigram: int,
    min_freq_trigram_plus: int
) -> List:
    """
    Filter phrase_records based on UI selections.
    
    This applies both automatic filters (kinds, patterns, frequencies)
    and manual selections (checkboxes in table).
    
    Args:
        phrase_records: Original list of PhraseRecord objects (individual occurrences)
        phrase_df: Aggregated DataFrame with 'selected' column showing user selections
        include_kinds: List of phrase kinds to include (NP, VP)
        include_patterns: List of patterns to include
        min_freq_unigram: Minimum frequency for 1-word phrases
        min_freq_bigram: Minimum frequency for 2-word phrases
        min_freq_trigram_plus: Minimum frequency for 3+ word phrases
        
    Returns:
        Filtered list of PhraseRecord objects
    """
    # Build set of selected (phrase, kind, pattern) tuples from DataFrame
    selected_phrases = set()
    
    for _, row in phrase_df.iterrows():
        # Check if this phrase passed all filters
        if not row['selected']:
            continue
            
        # Check kind filter
        if row['kind'] not in include_kinds:
            continue
            
        # Check pattern filter
        if row['pattern'] not in include_patterns:
            continue
            
        # Check frequency filters
        num_words = row['num_words']
        count = row['count']
        
        if num_words == 1 and count < min_freq_unigram:
            continue
        if num_words == 2 and count < min_freq_bigram:
            continue
        if num_words >= 3 and count < min_freq_trigram_plus:
            continue
            
        # This phrase passed all filters
        selected_phrases.add((row['phrase'], row['kind'], row['pattern']))
    
    # Filter original phrase_records based on selected phrases
    filtered = []
    for rec in phrase_records:
        key = (rec.phrase, rec.kind, rec.pattern)
        if key in selected_phrases:
            filtered.append(rec)
    
    return filtered


def generate_code_from_settings() -> str:
    """Generate Python code snippet from current settings."""
    params = st.session_state
    
    # Build pattern sets
    filter_patterns = params.get('filter_patterns_np', []) + params.get('filter_patterns_vp', [])
    patterns_str = '{' + ', '.join(f'"{p}"' for p in filter_patterns) + '}'
    kinds_str = '{' + ', '.join(f'"{k}"' for k in params['filter_kinds']) + '}'
    
    code = f'''import phrasetopicminer as ptm

# Load your documents
docs = ["Your document text here...", "Another document..."]

# ============================================================================
# PHASE 1: PHRASE MINING
# ============================================================================

miner = ptm.PhraseMiner(
    method="{params['method']}",
    spacy_model="{params['spacy_model']}",
    include_verb_phrases={params['include_verb_phrases']},
    clean_markdown={params['clean_markdown']}
)

np_counter, vp_counters, phrase_records, sentences_by_doc = miner.mine_phrases_with_types(docs)
print(f"Mined {{len(phrase_records)}} phrases")

# ============================================================================
# PHASE 2: PHRASE FILTERING
# ============================================================================

# Apply automatic filters
filtered_phrases = [
    p for p in phrase_records
    if (
        # Kind filter
        p.kind in {kinds_str}
        # Pattern filter
        and p.pattern in {patterns_str}
        # Frequency filters
        and (
            (len(p.phrase.split()) == 1 and p.count >= {params['min_freq_unigram']})
            or (len(p.phrase.split()) == 2 and p.count >= {params['min_freq_bigram']})
            or (len(p.phrase.split()) >= 3 and p.count >= {params['min_freq_trigram_plus']})
        )
    )
]

# Apply manual selection (if you filtered specific phrases in the UI)
# selected_indices = [0, 1, 2, ...]  # Your manual selections from the table
# filtered_phrases = [filtered_phrases[i] for i in selected_indices]

print(f"Filtered to {{len(filtered_phrases)}} phrases")

# ============================================================================
# PHASE 3: TOPIC MODELING
# ============================================================================

modeler = ptm.TopicModeler(
    embedding_backend="{params['embedding_backend']}",
    embedding_model="{params['embedding_model']}",
    random_state=42
)

core_result = modeler.fit_core(
    # Pre-filtered phrases
    phrase_records=filtered_phrases,
    sentences_by_doc=sentences_by_doc,
    
    # Dimensionality Reduction
    pca_n_components={params['pca_n_components']},
    cluster_geometry="{params['cluster_geometry']}",
    umap_n_neighbors={params['umap_n_neighbors']},
    umap_min_dist={params['umap_min_dist']},
    umap_cluster_n_components={params['umap_cluster_n_components']},
    
    # Clustering
    clustering_algorithm="{params['clustering_algorithm']}",
    hdbscan_min_cluster_size={params['hdbscan_min_cluster_size']},
    hdbscan_min_samples={params['hdbscan_min_samples']},
    hdbscan_metric="{params['hdbscan_metric']}",
    kmeans_max_clusters={params['kmeans_max_clusters']},
    
    # Visualization
    viz_reducer="{params['viz_reducer']}",
    tsne_perplexity={params['tsne_perplexity']},
    tsne_learning_rate={params['tsne_learning_rate']},
    tsne_n_iter={params['tsne_n_iter']},
    
    # Representatives
    top_n_representatives={params['top_n_representatives']},
    
    verbose=True
)

# Build timeline
timeline_builder = ptm.TopicTimelineBuilder(
    timeline_mode="{params['timeline_mode']}",
    speech_rate_wpm={params['speech_rate_wpm']}
)
timeline_result = timeline_builder.build(core_result, sentences_by_doc)

print("Total NP phrases:", len(np_counter))
print("Total VP phrases:", sum(len(v) for v in vp_counters.values()))

print(f"Discovered {{len(core_result.clusters)}} topics")
print(core_result.phrases_df.head())
'''
    return code


def download_dataframe_as_csv(df: pd.DataFrame, filename: str):
    """Create download button for DataFrame as CSV."""
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Download {filename}",
        data=csv,
        file_name=filename,
        mime='text/csv'
    )


def download_plotly_as_html(fig, filename: str):
    """Create download button for Plotly figure as HTML."""
    buffer = io.StringIO()
    fig.write_html(buffer)
    html_bytes = buffer.getvalue().encode()
    st.download_button(
        label=f"📥 Download {filename}",
        data=html_bytes,
        file_name=filename,
        mime='text/html',
        use_container_width=True
    )

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    initialize_session_state()
    
    # ========================================================================
    # HEADER & PROGRESS
    # ========================================================================
    
    st.title("🔬 PhraseTopicMiner Playground")
    st.markdown("""
    An interactive dashboard to explore **phrase-centric topic modeling** with the PhraseTopicMiner library.
    Educational tool for researchers and developers to understand PTM's capabilities.
    """)

    # Show pipeline progress
    render_progress_tracker()

    # ========================================================================
    # SIDEBAR: CONFIGURATION
    # ========================================================================
    
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.markdown("""
        Configure parameters for each phase below. Changes take effect when you click the respective action button in the main area.
        """)
        
        st.markdown("---")
        
        # --------------------------------------------------------------------
        # PHRASE MINING CONFIGURATION
        # --------------------------------------------------------------------
        
        with st.expander("⛏️ **Phase 2: Mining**", expanded=False):
            # st.markdown("**POS Tagging:**")
            st.segmented_control(
                "**POS-tagging Method**",
                ["spacy", "nltk"],
                selection_mode="single",
                default="spacy",
                key="method",
                help="Method for part-of-speech tagging: **spaCy** provides better accuracy; **NLTK** is faster",
                # label_visibility="collapsed"
            )
            
            method_val = st.session_state.method
            if method_val == "spacy":
                st.selectbox(
                    "spaCy Model",
                    ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"],
                    index=2,
                    key="spacy_model",
                    help="Larger models are more accurate but slower"
                )

            st.markdown("**Options:**")
            st.checkbox(
                "Include Verb Phrases",
                key="include_verb_phrases",
                help="Extract verb phrases in addition to noun phrases"
            )
            
            st.checkbox(
                "Clean Markdown",
                key="clean_markdown",
                help="Remove markdown syntax (**, ##, etc.) from text"
            )
        
        # --------------------------------------------------------------------
        # TOPIC MODELING CONFIGURATION
        # --------------------------------------------------------------------
        
        with st.expander("🧩 **Phase 4: Modeling**", expanded=False):
            
            st.markdown("**Embeddings:**")
            embeddings_help = """
                ## ℹ️ What are embeddings?
                **Embeddings** convert text into numerical vectors that capture semantic meaning.
                
                - Phrases with similar meanings have similar vectors
                - Essential for clustering and finding topics
                - Higher-quality embeddings → better topic coherence
                
                **Recommended:** Use `sentence_transformers` with `all-MiniLM-L6-v2` for best balance of speed and quality.
                """
            st.selectbox(
                "Embedding Backend",
                ["sentence_transformers", "spacy", "custom"],
                key="embedding_backend",
                help=embeddings_help
            )
            
            if st.session_state.embedding_backend == "sentence_transformers":
                st.selectbox(
                    "Embedding Model",
                    ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L6-v2"],
                    key="embedding_model",
                    help=embeddings_help
                )
            
            st.markdown("---")
            st.markdown("**Dimensionality Reduction:**")
            dim_reduction_help = """
                ## ℹ️ Why reduce dimensions?
                **PCA (Optional Denoising):**
                - Removes noise from high-dimensional embeddings
                - Projects to lower dimensions (e.g., 50D)
                - Stabilizes clustering, speeds up UMAP
                - Set to 0 to disable
                
                **Recommended:** PCA=50, UMAP=umap_nd (10D)
                
                """
            
            st.slider(
                "PCA Components (0 = disabled)",
                min_value=0, max_value=100, value=30,
                key="pca_n_components",
                help=dim_reduction_help
            )

            umap_help = """
                **UMAP (Manifold Learning):**
                - Further reduces dimensions for clustering
                - Preserves both local and global structure
                - `umap_nd`: Use 10D for clustering (recommended)
                - `umap_2d`: Use 2D directly (faster but less accurate)
                - **N Neighbors** (`umap_n_neighbors`): Larger = more global structure
                - `umap_min_dist`: **Minimum distance** between points in the embedding. Smaller = tighter clusters (cluster compactness)
                - **Cluster N Components** (`umap_cluster_n_components`): Target dimensions for clustering (if using `umap_nd`)
                
                **Recommended:** PCA=50, UMAP=umap_nd (10D)
                """

            st.markdown("---")
            st.markdown("**UMAP (Manifold Learning):**")
            
            st.segmented_control(
                "Cluster Geometry",
                ["umap_nd", "umap_2d"],
                selection_mode="single",
                default="umap_nd",
                key="cluster_geometry",
                help="**umap_2d**: Cluster in 2D space (faster but less accurate). **umap_nd**: Cluster in higher dimensional space."
            )
            
            with st.expander("UMAP Settings"):
                st.slider(
                    "N Neighbors",
                    min_value=2, max_value=50, value=10,
                    key="umap_n_neighbors",
                    help=umap_help
                )
                
                st.slider(
                    "Minimum Distance",
                    min_value=0.0, max_value=1.0, value=0.05, step=0.01,
                    key="umap_min_dist",
                    help=umap_help
                )

                if st.session_state.cluster_geometry == "umap_nd":
                    st.slider(
                        "UMAP Cluster N Components",
                        min_value=2, max_value=50, value=10,
                        key="umap_cluster_n_components",
                        help=umap_help
                    )
            
            st.markdown("---")
            st.markdown("**Clustering:**")

            hdbscan_help="""
                ## ℹ️ How does clustering work?
                
                **HDBSCAN (Density-Based):**
                - Finds clusters of varying density
                - Automatically determines number of clusters
                - Can mark outliers as "noise" (cluster -1)
                - **Min Cluster Size:** Minimum phrases per cluster
                - **Min Samples:** Core point threshold
                - **Metric:** Distance measure (euclidean recommended)
               
                **Recommended:** HDBSCAN with min_cluster_size=10 for most cases
            """

            kmeans_help="""
                **KMeans (Centroid-Based):**
                - Partitions data into K clusters
                - Auto-K uses Silhouette to find best K
                - Every phrase assigned to a cluster (no noise)
                - **Max Clusters:** Upper bound for auto-K search
                
                **Recommended:** HDBSCAN with min_cluster_size=10 for most cases
            """
            
            st.segmented_control(
                "Algorithm",
                ["hdbscan", "kmeans"],
                selection_mode="single",
                default="hdbscan",
                key="clustering_algorithm",
                help="**HDBSCAN**: Density-based, finds natural clusters of varying density and can mark outliers as 'noise' (cluster -1). **KMeans** Centroid-Based, partitions data into a given number of clusters (k). Auto-K uses Silhouette to find best K."
            )
            
            clustering_val = st.session_state.clustering_algorithm
            
            if clustering_val == "hdbscan":
                st.slider(
                    "Min Cluster Size",
                    min_value=2, max_value=50, value=10,
                    key="hdbscan_min_cluster_size",
                    help=hdbscan_help
                )
                
                st.slider(
                    "Min Samples",
                    min_value=1, max_value=20, value=3,
                    key="hdbscan_min_samples",
                    help="HDBSCAN min_samples parameter (noise sensitivity)"
                )
                
                st.selectbox(
                    "Metric",
                    ["euclidean", "manhattan", "chebyshev", "minkowski", "hamming"],
                    key="hdbscan_metric",
                    help="Distance metric for HDBSCAN"
                )
            else:
                st.slider(
                    "Max Clusters",
                    min_value=2, max_value=50, value=30,
                    key="kmeans_max_clusters",
                    help=kmeans_help
                )
            
            st.markdown("---")
            with st.expander("**Visualization**"):
                st.segmented_control(
                    "2D Reducer",
                    ["umap_2d", "tsne_2d", "same"],
                    selection_mode="single",
                    default="umap_2d",
                    key="viz_reducer",
                    help="How to create 2D visualization coordinates. **umap_2d:** Use UMAP to create 2D projection (recommended). **tsne_2d:** Use t-SNE (slower, preserves local structure). **same:** Reuse clustering coordinates if already 2D"
                )
                
                viz_val = st.session_state.viz_reducer
                
                if viz_val == "tsne_2d":
                    st.slider(
                        "Perplexity",
                        min_value=5.0, max_value=50.0, value=30.0, step=1.0,
                        key="tsne_perplexity",
                        help="t-SNE perplexity parameter"
                    )
                    
                    st.slider(
                        "Learning Rate",
                        min_value=10.0, max_value=1000.0, value=200.0, step=10.0,
                        key="tsne_learning_rate",
                        help="t-SNE learning rate"
                    )
                    
                    st.slider(
                        "Iterations",
                        min_value=250, max_value=5000, value=1000, step=250,
                        key="tsne_n_iter",
                        help="Number of t-SNE iterations"
                    )
            
            st.markdown("---")
            st.slider(
                "Top N Representatives",
                min_value=5, max_value=50, value=30,
                key="top_n_representatives",
                help="Number of top phrases (by frequency) to include as `representative_phrases` in each TopicCluster"
            )
        
        # --------------------------------------------------------------------
        # TIMELINE CONFIGURATION
        # --------------------------------------------------------------------
        
        with st.expander("⏱️ **Timeline Settings**", expanded=False):
            st.segmented_control(
                "Timeline Mode",
                ["reading_time", "index"],
                selection_mode="single",
                default="reading_time",
                key="timeline_mode",
                help="How to construct the timeline"
            )
            
            timeline_val = st.session_state.timeline_mode
            
            if timeline_val == "reading_time":
                st.slider(
                    "Reading Speed (WPM)",
                    min_value=100, max_value=300, value=160,
                    key="speech_rate_wpm",
                    help="Words per minute for reading time estimation"
                )

        # --------------------------------------------------------------------
        # HELP & RESOURCES
        # --------------------------------------------------------------------
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666;">
            <p>PhraseTopicMiner</p>
            <p>
                <a href="https://github.com/towardsNLP/PhraseTopicMiner" target="_blank">GitHub</a> |
                <a href="https://pypi.org/project/phrasetopicminer/" target="_blank">PyPI</a>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ========================================================================
    # PHASE 1: DATA INPUT
    # ========================================================================
    
    st.markdown("### Phase 1: Data Input")
    
    # Determine if expander should be open:
    # - If data not ready, keep open
    # - If just switched data source, keep open (even if previous was ready)
    # - If data ready and no recent switch, collapse
    if st.session_state.data_source_just_switched:
        phase1_expanded = True
        st.session_state.data_source_just_switched = False  # Reset flag
    else:
        phase1_expanded = not st.session_state.data_ready
    
    with st.expander("📁 Select and submit your data", expanded=phase1_expanded):
        data_source = st.segmented_control(
            "Data source:",
            ["Use Example Corpus", "Paste Text", "Upload File(s)"],
            selection_mode="single",
            default="Use Example Corpus",
            key="data_source",
        )
        
        # Detect data source change
        if 'previous_data_source' not in st.session_state:
            st.session_state.previous_data_source = data_source
        elif st.session_state.previous_data_source != data_source:
            # Data source changed - clear everything and mark as switched
            st.session_state.phrase_records = None
            st.session_state.sentences_by_doc = None
            st.session_state.np_counter = None
            st.session_state.vp_counters = None
            st.session_state.phrase_df = None
            st.session_state.filtered_phrase_records = None
            st.session_state.mining_complete = False
            st.session_state.results = None
            st.session_state.labeling_result = None
            st.session_state.data_ready = False
            st.session_state.data_source_just_switched = True  # NEW: Set flag
            st.session_state.previous_data_source = data_source
            st.info(f"Switched to: **{data_source}**. Previous results cleared.")
            st.rerun()  # Rerun to expand the expander
        
        # Handle each data source
        if st.session_state.data_source == "Use Example Corpus":
            st.info("📚 Using 5 philosophical dialogues from Bryan Magee's interview series.")
            try:
                sample_data = load_sample_documents("examples/sample_corpus")
                # st.markdown(format_corpus_stats_for_display(sample_data['corpus_stats']))
                st.session_state.data_ready = True
            except Exception as e:
                st.warning(f"Could not load sample corpus: {e}")
                st.session_state.data_ready = False

        elif data_source == "Paste Text":
            with st.form(key="paste_text_form"):
                st.markdown("Paste your documents below. Separate multiple documents with `===` on a new line.")
                text_input = st.text_area(
                    "Text input:",
                    height=200,
                    placeholder="Document 1 text here...\n===\nDocument 2 text here...\n===\nDocument 3...",
                    label_visibility="collapsed"
                )
                
                submitted = st.form_submit_button("📤 Submit Data", type="primary", use_container_width=True)
                
                if submitted:
                    if text_input.strip():
                        # Clear cache
                        st.session_state.phrase_records = None
                        st.session_state.sentences_by_doc = None
                        st.session_state.np_counter = None
                        st.session_state.vp_counters = None
                        st.session_state.phrase_df = None
                        st.session_state.filtered_phrase_records = None
                        st.session_state.mining_complete = False
                        st.session_state.results = None
                        st.session_state.labeling_result = None
                        
                        # Store data
                        st.session_state.text_input = text_input
                        st.session_state.data_ready = True
                        st.success("✅ Text data submitted! Proceed to Phase 2 below.")
                        st.rerun()
                    else:
                        st.error("⚠️ Please enter some text before submitting.")
                        st.session_state.data_ready = False
            
            if st.session_state.data_ready and st.session_state.text_input:
                num_docs = len([p for p in st.session_state.text_input.split('===') if p.strip()])
                st.caption(f"📝 Loaded: {num_docs} document(s), {len(st.session_state.text_input)} characters")
                
        elif data_source == "Upload File(s)":
            with st.form(key="upload_files_form"):
                st.markdown("Upload one or more `.txt` files. Each file will be treated as a separate document.")
                uploaded_files = st.file_uploader(
                    "Files:",
                    type=["txt"],
                    accept_multiple_files=True,
                    label_visibility="collapsed"
                )
                
                submitted = st.form_submit_button("📤 Submit Data", type="primary", use_container_width=True)
                
                if submitted:
                    if uploaded_files:
                        # Clear cache
                        st.session_state.phrase_records = None
                        st.session_state.sentences_by_doc = None
                        st.session_state.np_counter = None
                        st.session_state.vp_counters = None
                        st.session_state.phrase_df = None
                        st.session_state.filtered_phrase_records = None
                        st.session_state.mining_complete = False
                        st.session_state.results = None
                        st.session_state.labeling_result = None
                        
                        # Process files
                        docs = []
                        for uploaded_file in uploaded_files:
                            try:
                                text = uploaded_file.read().decode("utf-8", errors='ignore')
                                docs.append(text)
                            except Exception as e:
                                st.error(f"Error reading {uploaded_file.name}: {e}")
                        
                        st.session_state.uploaded_docs = docs
                        st.session_state.data_ready = True
                        st.success(f"✅ {len(docs)} file(s) submitted! Proceed to Phase 2 below.")
                        st.rerun()
                    else:
                        st.error("⚠️ Please upload at least one file.")
                        st.session_state.data_ready = False
            
            if st.session_state.data_ready and st.session_state.uploaded_docs:
                st.caption(f"📁 Loaded: {len(st.session_state.uploaded_docs)} file(s)")
    
    
    # ========================================================================
    # PHASE 2: PHRASE MINING
    # ========================================================================
    
    st.markdown("---")
    st.markdown("### Phase 2: Phrase Mining")
    
    if not st.session_state.data_ready:
        st.info("👆 Complete Phase 1 above to proceed")
    else:
        st.markdown("""
        Extract noun phrases (NPs) and optionally verb phrases (VPs) using POS patterns.
        👈 Configure mining parameters in the sidebar, then click the button below.
        """)
        
        if st.button("⛏️ **Mine Phrases**", type="primary", use_container_width=True, key="mine_button"):
            docs = get_docs_from_input()
            
            if not docs:
                st.error("⚠️ No data available. Please check your input.")
            else:
                # Build current config
                current_config = {
                    'method': st.session_state.method,
                    'spacy_model': st.session_state.spacy_model,
                    'include_verb_phrases': st.session_state.include_verb_phrases,
                    'clean_markdown': st.session_state.clean_markdown,
                }

                # Check if config changed
                if st.session_state.last_mining_config != current_config:
                    # Config changed - clear cache
                    st.session_state.phrase_records = None
                    st.session_state.sentences_by_doc = None
                    st.session_state.np_counter = None
                    st.session_state.vp_counters = None
                    st.session_state.phrase_df = None
                    st.session_state.filtered_phrase_records = None
                    st.session_state.mining_complete = False
                    st.session_state.results = None
                    st.session_state.labeling_result = None
                    st.session_state.last_mining_config = current_config
        
                # Extract mining params
                method_val = st.session_state.method
                
                # Load miner
                with st.spinner("Loading phrase miner..."):
                    miner = load_phrase_miner(
                        method=method_val,
                        spacy_model=st.session_state.spacy_model,
                        include_verb_phrases=st.session_state.include_verb_phrases,
                        clean_markdown=st.session_state.clean_markdown
                    )
                
                # Run mining
                result = run_phrase_mining(miner, docs, current_config)
                
                if result:
                    np_counter, vp_counters, phrase_records, sentences_by_doc = result
                    st.session_state.np_counter = np_counter
                    st.session_state.vp_counters = vp_counters
                    st.session_state.phrase_records = phrase_records
                    st.session_state.sentences_by_doc = sentences_by_doc
                    st.session_state.mining_complete = True
                    
                    # Show success
                    total_np = len(np_counter)
                    total_vp = sum(len(v) for v in vp_counters.values())
                    st.success(f"✅ Mined **{total_np} NP + {total_vp} VP = {total_np + total_vp}** unique phrases ({len(phrase_records)} occurrences)")
                    st.rerun()
    
    # ========================================================================
    # PHASE 3: PHRASE FILTERING
    # ========================================================================
    
    if st.session_state.mining_complete and st.session_state.phrase_records:
        
        st.markdown("---")
        st.markdown("### Phase 3: Phrase Filtering")
        
        # Collapse if complete
        phase3_expanded = not st.session_state.results

        with st.expander("🗂️ Review and filter phrases", expanded=phase3_expanded):
                    
            # Build DataFrame if not exists
            if st.session_state.phrase_df is None:
                st.session_state.phrase_df = build_phrase_dataframe(st.session_state.phrase_records)
            
            phrase_df = st.session_state.phrase_df
            
            # Statistics - styled cards
            if st.session_state.np_counter:
                total_np = len(st.session_state.np_counter)
                if st.session_state.vp_counters:
                    total_vp = sum(len(v) for v in st.session_state.vp_counters.values())
                else:
                    total_vp = 0
                total_unique = total_np + total_vp
                total_occurrences = len(st.session_state.phrase_records)
                
                # Mining Statistics - Simple and clean
                st.markdown("**Mining Results:**")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("NP Phrases", total_np, border=True)
                with col2:
                    st.metric("VP Phrases", total_vp, border=True)
                with col3:
                    st.metric("Total Unique", total_unique, border=True)
                with col4:
                    st.metric("Occurrences", total_occurrences, border=True)
                
                st.markdown("---")
                
                st.markdown("<br>", unsafe_allow_html=True)
            
            
            # Filtering Options
            st.markdown("**Filtering Options:**")
            
            # Pattern guide
            with st.expander("📚 Pattern Guide - Learn about NP and VP patterns"):
                st.markdown("""
                ### Noun Phrase (NP) Patterns
                
                **NPs are the primary carriers of meaning in topic modeling.**
                
                #### BaseNP – `(A|N)* N`
                Base noun phrase: optional adjectives/nouns followed by a noun head
                - *Examples:* `neural networks`, `deep learning`, `topic models`
                
                #### NP+PP – `BaseNP + PP`  
                Noun phrase with one prepositional phrase (PP = `P D* (A|N)* N`)
                - *Examples:* `model of computation`, `theory of mind`, `analysis of variance`
                
                #### NP+multiPP – `BaseNP + PP + PP+`
                Noun phrase with multiple prepositional phrases
                - *Examples:* `impact of AI on society`, `analysis of variance in statistics`
                
                ---
                
                ### Verb Phrase (VP) Patterns
                
                **VPs capture actions and states - optional but useful for process-oriented text.**
                
                #### VerbObj – `V (A|N)* N`
                Verb followed by object
                - *Examples:* `train models`, `analyze data`, `optimize performance`
                
                #### VerbPP – `V + PP`
                Verb followed by prepositional phrase
                - *Examples:* `move to production`, `look at results`, `run on hardware`
                
                #### SubjVerb – `N + V + ...`
                Subject followed by non-copular verb (action verbs)
                - *Examples:* `researchers demonstrate`, `students write`, `systems process`
                
                #### SubjCopula – `N + be + ...`
                Subject followed by copular "be" verb (state/property)
                - *Examples:* `model is unstable`, `results are promising`, `approach is novel`
                
                ---
                
                **Tag Alphabet:**  
                `N`=noun, `V`=verb, `A`=adjective, `P`=preposition, `D`=determiner
                """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.session_state.include_verb_phrases:
                    default_options=["NP", "VP"]
                else:
                    default_options=["NP"]
                selected_kinds = st.pills(
                    "Phrase Kinds",
                    default_options,
                    selection_mode="multi",
                    default=default_options,
                    key="filter_kinds",
                    help="NP=Noun Phrases (concepts), VP=Verb Phrases (actions)"
                )
            
            with col2:
                selected_patterns_np = st.pills(
                    "NP Patterns",
                    ["BaseNP", "NP+PP", "NP+multiPP"],
                    selection_mode="multi",
                    default=["BaseNP", "NP+PP", "NP+multiPP"],
                    key="filter_patterns_np",
                )
            
            if st.session_state.include_verb_phrases and "VP" in selected_kinds:
                selected_patterns_vp = st.pills(
                    "VP Patterns",
                    ["VerbObj", "VerbPP", "SubjVerb", "SubjCopula"],
                    selection_mode="multi",
                    default=["VerbObj"],
                    key="filter_patterns_vp",
                )
            else:
                selected_patterns_vp = []
            
            # Frequency filters
            st.markdown("**Minimum Frequency Thresholds:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                min_freq_1 = st.number_input(
                    "1-word phrases",
                    min_value=1, value=3, step=1,
                    key="min_freq_unigram"
                )
            with col2:
                min_freq_2 = st.number_input(
                    "2-word phrases",
                    min_value=1, value=2, step=1,
                    key="min_freq_bigram"
                )
            with col3:
                min_freq_3 = st.number_input(
                    "3+ word phrases",
                    min_value=1, value=1, step=1,
                    key="min_freq_trigram_plus"
                )
            
            # Apply filters
            selected_patterns = list(set(selected_patterns_np + selected_patterns_vp))
            
            filtered_mask = (
                phrase_df['kind'].isin(selected_kinds) &
                phrase_df['pattern'].isin(selected_patterns) &
                (
                    ((phrase_df['num_words'] == 1) & (phrase_df['count'] >= min_freq_1)) |
                    ((phrase_df['num_words'] == 2) & (phrase_df['count'] >= min_freq_2)) |
                    ((phrase_df['num_words'] >= 3) & (phrase_df['count'] >= min_freq_3))
                )
            )
            
            display_df = phrase_df[filtered_mask].copy()
            
            # Search functionality
            st.markdown("---")
            # Search functionality
            st.markdown("**Search & Select:**")
            
            col1, col2 = st.columns([5, 1])
            with col1:
                search_term = st.text_input(
                    "Search phrases:",
                    placeholder="Type to find specific phrases...",
                    key="phrase_search",
                    label_visibility="collapsed"
                )
    
            with col2:
                def clear_phrase_search():
                    st.session_state.phrase_search = ""
                if search_term: 
                    st.button("Clear Search", key="clear_search", on_click=clear_phrase_search)
                    
            
            # Apply search
            if search_term:
                search_mask = display_df['phrase'].str.contains(search_term, case=False, na=False)
                display_df = display_df[search_mask]
                st.info(f"Found {len(display_df)} phrases matching '{search_term}'")
            
            # Bulk selection
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("✓ Select All Visible", use_container_width=True):
                    st.session_state.phrase_df.loc[display_df.index, 'selected'] = True
                    st.rerun()
            with col2:
                if st.button("✕ Deselect All Visible", use_container_width=True):
                    st.session_state.phrase_df.loc[display_df.index, 'selected'] = False
                    st.rerun()
            
            # Show table
            if len(display_df) == 0:
                st.warning("⚠️ No phrases match the current filters. Try relaxing the constraints.")
            else:
                # Get total unique count
                if st.session_state.np_counter and st.session_state.vp_counters:
                    total_unique = len(st.session_state.np_counter) + sum(len(v) for v in st.session_state.vp_counters.values())
                else:
                    total_unique = len(phrase_df)
                
                st.info(f"📋 Showing {len(display_df)} / {total_unique} phrases")
                
                # Interactive table
                edited_df = st.data_editor(
                    display_df,
                    use_container_width=True,
                    num_rows="fixed",
                    height=400,
                    column_config={
                        "selected": st.column_config.CheckboxColumn(
                            "✓",
                            help="Include in topic modeling",
                            default=True,
                            width="small"
                        ),
                        "phrase": st.column_config.TextColumn("Phrase", width="large"),
                        "count": st.column_config.NumberColumn("Count", format="%d", width="small"),
                        "kind": st.column_config.TextColumn("Kind", width="small"),
                        "pattern": st.column_config.TextColumn("Pattern", width="small"),
                        "num_words": st.column_config.NumberColumn("Words", format="%d", width="small"),
                    },
                    hide_index=False,
                    column_order=["selected", "phrase", "count", "kind", "pattern", "num_words"],
                )
                
                # Update session state
                st.session_state.phrase_df.loc[edited_df.index, 'selected'] = edited_df['selected']
                
                # Calculate stats
                filtered_and_selected = display_df[edited_df['selected']].index
                num_will_use = len(filtered_and_selected)
                num_total_displayed = len(display_df)
                
                if st.session_state.np_counter and st.session_state.vp_counters:
                    total_unique = len(st.session_state.np_counter) + sum(len(v) for v in st.session_state.vp_counters.values())
                else:
                    total_unique = len(phrase_df)
                
                st.progress(num_will_use / total_unique if total_unique > 0 else 0)
                st.write(f"**Will use for modeling:** {num_will_use} / {total_unique} phrases ({num_will_use/total_unique*100:.1f}%)")
                
                # Export options
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📥 Export Filtered Phrases", use_container_width=True):
                        csv = display_df.to_csv(index=True)
                        st.download_button(
                            "Download CSV",
                            csv,
                            "filtered_phrases.csv",
                            "text/csv",
                            key="download_filtered"
                        )
                with col2:
                    if st.button("📥 Export Selected Phrases", use_container_width=True):
                        selected_phrases = display_df[edited_df['selected']]
                        csv = selected_phrases.to_csv(index=True)
                        st.download_button(
                            "Download CSV",
                            csv,
                            "selected_phrases.csv",
                            "text/csv",
                            key="download_selected"
                        )
    
    elif not st.session_state.mining_complete:
        pass  # Message already shown in Phase 2
    
    # ========================================================================
    # PHASE 4: TOPIC MODELING
    # ========================================================================
    
    if st.session_state.mining_complete and st.session_state.phrase_records:
        
        st.markdown("---")
        st.markdown("### Phase 4: Topic Modeling")
        st.markdown("""
        Build topic clusters using embeddings, dimensionality reduction, and clustering.
        👈 Configure modeling parameters in the sidebar, then click the button below.
        """)
        
        if st.button("🧩 **Build Topics**", type="primary", use_container_width=True):
            # Get filtering params
            selected_patterns = list(set(
                st.session_state.get('filter_patterns_np', []) + 
                st.session_state.get('filter_patterns_vp', [])
            ))
      
            # Filter phrase_records
            filtered_phrase_records = filter_phrase_records(
                phrase_records=st.session_state.phrase_records,
                phrase_df=st.session_state.phrase_df,
                include_kinds=st.session_state.filter_kinds,
                include_patterns=selected_patterns,
                min_freq_unigram=st.session_state.min_freq_unigram,
                min_freq_bigram=st.session_state.min_freq_bigram,
                min_freq_trigram_plus=st.session_state.min_freq_trigram_plus
            )
            
            if not filtered_phrase_records:
                st.error("⚠️ No phrases selected. Please select at least some phrases in the filtering table.")
            else:
                # Build parameters dict (values already clean, no lists to extract)
                params = {k: v for k, v in st.session_state.items()}
                
                # Load modeler
                with st.spinner("Loading topic modeler..."):
                    modeler = load_topic_modeler(
                        embedding_backend=params['embedding_backend'],
                        embedding_model=params['embedding_model']
                    )
                
                # Run topic modeling
                st.session_state.results = run_topic_modeling(
                    modeler,
                    filtered_phrase_records,
                    st.session_state.sentences_by_doc,
                    params
                )
                
                if st.session_state.results:
                    st.session_state.labeling_result = None
                    st.session_state.selected_cluster_id = None
                    st.success("✅ Topics built successfully! Scroll down to explore results.")
                    st.success(f"✅ Found **{len(st.session_state.results['core_result'].clusters)}** topic clusters!")
                    st.rerun()

    # ========================================================================
    # RESULTS DISPLAY
    # ========================================================================
    
    if not st.session_state.results:
        return
    
    st.markdown("---")
    st.markdown("## 📋 Results")
    
    # Unpack results
    results = st.session_state.results
    core = results['core_result']
    timeline = results['timeline_result']
    sents_by_doc = results['sentences_by_doc']
    labels = st.session_state.labeling_result
    
    # Overall summary
    st.markdown(f"**{len(core.clusters)} topic clusters identified**")
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        total_phrases = len(core.phrases_df)
        st.metric("Total Phrases", f"{total_phrases:,}", border=True)
    with col2:
        total_occurrences = int(core.phrases_df['count'].sum())
        st.metric("Total Occurrences", f"{total_occurrences:,}", border=True)
    with col3:
        non_noise = [c for c in core.clusters if c.cluster_id != -1]
        st.metric("Topics (excluding noise)", len(non_noise), border=True)
    
    st.markdown('---')

    # ========================================================================
    # TOPIC EXPANDERS (Main Content)
    # ========================================================================
    
    st.markdown("### Topic Details")
    st.markdown("Click on each topic below to explore phrases and sentences.")
    
    # Sort clusters by importance (noise last)
    sorted_clusters = sorted(
        [c for c in core.clusters if c.cluster_id != -1],
        key=lambda x: x.importance_score,
        reverse=True
    )
    
    # Add noise cluster at the end if it exists
    noise_clusters = [c for c in core.clusters if c.cluster_id == -1]
    sorted_clusters.extend(noise_clusters)


    for cluster in sorted_clusters:
        cluster_id = cluster.cluster_id

        # Get phrase dataframe to compute centrality-sorted list
        phrase_df = create_cluster_phrases_dataframe(
            core_result=core,
            cluster_id=cluster_id,
            timeline_result=timeline,
            sentences_by_doc=sents_by_doc,
            include_occurrences=True
        )
        
        # Sort by centrality (ascending - lower values = more central)
        centrality_sorted_phrases = phrase_df.nsmallest(len(phrase_df), 'centrality')['phrase'].tolist()
        centrality_sorted_counts = phrase_df.nsmallest(len(phrase_df), 'centrality')['count'].tolist()

        
        # Expander title with top 3 most central phrases
        if cluster_id == -1:
            title = f"**Noise (Cluster -1)** — {', '.join(centrality_sorted_phrases[:3])}"
        else:
            title = f"**Topic {cluster_id}** — {', '.join(centrality_sorted_phrases[:3])}"

        with st.expander(title, expanded=False):
            
            # === HEADER: Summary Stats ===
            st.markdown(f"- ###### **Size:** `{len(cluster.phrases)}` unique phrases | `{cluster.total_count}` phrase occurrences ")
            
            # Show phrase list sorted by centrality with counts
            phrase_list = []
            for i in range(len(centrality_sorted_phrases)):
                phrase_list.append(f"{centrality_sorted_phrases[i]} `({centrality_sorted_counts[i]})`")
            st.markdown(f"- **Phrases** (sorted by centrality):")
            st.markdown(f"&nbsp;&nbsp;&emsp;{', '.join(phrase_list)}")
            
            st.markdown("---")
            
            # === ROW 1: Comprehensive Phrases DataFrame ===
            st.markdown("#### 📋 Cluster Phrases")
            st.markdown("Complete information for all phrases in this cluster.")
            
            try:
                
                # Display table (without occurrences column for cleaner view)
                display_cols = ['phrase', 'count', 'kind','pattern','num_words', 
                                'importance', 'freq_len_norm', 'centrality', 'centrality_norm', 
                                'num_docs', 'occurrences']
                st.dataframe(
                    phrase_df[display_cols],
                    use_container_width=True,
                    height=400
                )
                
                # Standard CSV with occurrences column
                csv_standard = phrase_df.to_csv(index=False)
                st.download_button(
                    f"📥 Export Cluster {cluster_id} Phrases",
                    csv_standard,
                    f"cluster_{cluster_id}_phrases.csv",
                    "text/csv",
                    key=f"download_standard_{cluster_id}",
                    use_container_width=True
                )
            
            except Exception as e:
                st.error(f"Could not create phrases dataframe: {e}")
                import traceback
                st.code(traceback.format_exc())
            
            st.markdown("---")

            # === ROW 2: Cluster Sentences DataFrame (NEW) ===
            st.markdown("#### 📝 Cluster Sentences")
            st.markdown("All sentences in this cluster, in reading order.")
            
            try:
                sentences_df = create_cluster_sentences_dataframe(
                    core_result=core,
                    timeline_result=timeline,
                    cluster_id=cluster_id,
                    sentences_by_doc=sents_by_doc
                )
                
                if not sentences_df.empty:
                    st.dataframe(sentences_df, use_container_width=True, height=400)
                    
                    # Export sentences
                    csv_sentences = sentences_df.to_csv(index=False)
                    st.download_button(
                        f"📥 Export Cluster {cluster_id} Sentences",
                        csv_sentences,
                        f"cluster_{cluster_id}_sentences.csv",
                        "text/csv",
                        key=f"download_sentences_{cluster_id}",
                        use_container_width=True
                    )
                else:
                    st.info("No sentences found for this cluster.")
            
            except Exception as e:
                st.warning(f"Could not create sentences dataframe: {e}")
                import traceback
                st.code(traceback.format_exc())
            

    
    # ========================================================================
    # PHASE 5: LLM TOPIC LABELING (OPTIONAL)
    # ========================================================================

    # LLM Labeling (Optional)
    st.markdown("---")
    st.markdown("### Phase 5: Generate Labels (Optional)")

    with st.expander("🏷️ Use GPT to generate human-friendly labels", expanded=False):
        st.markdown("Optionally use OpenAI's GPT models to generate interpretable labels for each topic.")
        
        col1, col2 = st.columns(2)
        with col1:
            api_key = st.text_input(
                "OpenAI API Key", 
                type="password", 
                key="api_key_input", 
                help="Your OpenAI API key (starts with 'sk-')"
            )

            if api_key == PASSWORD:
                api_key = OPENAI_API_KEY
        with col2:
            llm_model = st.selectbox("LLM Model", ["gpt-4o-mini", "gpt-4-turbo", "gpt-4o"], key="llm_model_select")
        
        if st.button(" 🏷️ Generate Labels", use_container_width=True):
            if api_key:
                with st.spinner("Generating topic labels with GPT..."):
                    try:
                        # Call your labeling function
                        st.session_state.labeling_result = run_llm_labeling(core, sents_by_doc, api_key, llm_model)
                        labels = st.session_state.labeling_result
                        if labels:
                            st.success("✅ Topic labels generated!")
                            st.rerun()
                    except Exception as e:
                            st.error(f"Error generating labels: {e}")
            else:
                st.error("Please enter your OpenAI API Key.")

        # Show labeled results
        if st.session_state.labeling_result:
            st.markdown("---")
            st.markdown("### 🏷️ Labeled Topics")
           
            for cluster in st.session_state.labeling_result.labeled_clusters:
                # for cluster_id, label in item:
                if cluster.cluster_id == -1:
                    continue
                st.markdown(f"**Topic {cluster.cluster_id}:** {cluster.label.title}")
                st.caption(cluster.label.description)


    # ========================================================================
    # GENERATE CENTRALITY-BASED LABELS FOR VISUALIZATIONS
    # ========================================================================
    
    # Create cluster_name_map from top 2 central phrases
    cluster_name_map = {}
    
    for cluster in core.clusters:
        # Get phrases sorted by centrality
        phrase_df = create_cluster_phrases_dataframe(
            core_result=core,
            cluster_id=cluster.cluster_id,
            timeline_result=timeline,
            sentences_by_doc=sents_by_doc,
            include_occurrences=False
        )
        
        if not phrase_df.empty:
            centrality_sorted = phrase_df.nsmallest(2, 'centrality')['phrase'].tolist()
            cluster_name_map[cluster.cluster_id] = ', '.join(centrality_sorted)
            cluster_name_map[cluster.cluster_id] = f"{cluster.cluster_id}. {cluster_name_map[cluster.cluster_id]}"
        
            
    # ========================================================================
    # GLOBAL VISUALIZATIONS
    # ========================================================================
    
    st.markdown("---")
    st.markdown("### 🗺️ Global Visualizations")
    
    # Topic Maps (Interactive + Static with Tabs)
    with st.expander("🗺️ **Topic Maps**", expanded=True):
        st.markdown("Visualize all phrases in 2D semantic space.")
        
        tab1, tab2 = st.tabs([ "🖼️ Static", "📍 Interactive"])
                
        with tab1:
            st.markdown("**Static Map** - High-resolution image for publication.")
            
            try:
                # Add cluster IDs to cluster names
                if labels and labels.cluster_name_map:
                    cluster_name_map = {
                        cid: f"{cid}. {name}" 
                        for cid, name in labels.cluster_name_map.items()
                    }
                
                fig_static, ax = ptm.make_datamapplot_static(
                    core,
                    cluster_name_map=cluster_name_map,
                    save_path="topic_map.png",
                    label_font_size=11,
                    use_medoids=True,
                )
                
                st.pyplot(fig_static)
                plt.close(fig_static)
                
                # Download button
                buf = io.BytesIO()
                fig_static.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                st.download_button(
                    label="📥 Download Static Map (PNG)",
                    data=buf.getvalue(),
                    file_name="topic_map.png",
                    mime="image/png",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Could not generate static map: {e}")

        with tab2:
            st.markdown("**Interactive Map** - Hover over points to see phrases and sentences.")
            
            try:
                # Add cluster IDs to cluster names
                if labels and labels.cluster_name_map:
                    cluster_name_map = {
                        cid: f"{cid}. {name}" 
                        for cid, name in labels.cluster_name_map.items()
                    }
                
                fig_interactive = ptm.make_datamapplot_interactive(
                    core,
                    sentences_by_doc=sents_by_doc,
                    cluster_name_map=cluster_name_map,
                    point_size=5,
                    title="Interactive Topic Map",
                    sub_title="Phrases clustered in 2D semantic space",
                )
                
                with tempfile.NamedTemporaryFile(mode='w+', suffix='.html', delete=False, encoding='utf-8') as tmp_file:
                    fig_interactive.save(tmp_file.name)
                    html_str = tmp_file.read()
                os.unlink(tmp_file.name)
                
                st.components.v1.html(html_str, height=800, scrolling=False)
                
            except Exception as e:
                st.error(f"Could not generate interactive map: {e}")

        
    # ========================================================================
    # CORPUS THEMATIC FLOW
    # ========================================================================
    
    with st.expander("🌊 **Corpus Thematic Flow**", expanded=True):
        st.markdown("Visualize how topics flow through your corpus over time/sequence (one dot per key sentence).")

        try:
            # Add cluster IDs to cluster names
            if labels and labels.cluster_name_map:
                cluster_name_map = {
                    cid: f"{cid}. {name}" 
                    for cid, name in labels.cluster_name_map.items()
                }
            
            doc_topic_profile = create_document_topic_profile(timeline_result=timeline)

            col1, col2 = st.columns(2)
            with col1:
                all_doc_ids = sorted(timeline.sentence_df['doc_index'].unique())
                selected_docs = st.multiselect(
                    "Filter by Document(s)", 
                    options=all_doc_ids, 
                    default=None
                )
                show_separators = st.checkbox("Show Document Separators", value=True)
            with col2:
                max_to_color = min(100, len(core.phrase_occurrences))
                top_n = st.slider("Number of Phrases to Color", 5, max_to_color, round(max_to_color/2), 5)
          
            corpus_flow_fig, corpus_flow = plot_corpus_thematic_flow(
                doc_topic_profile=doc_topic_profile,
                timeline_result=timeline,
                core_result=core,
                cluster_name_map=cluster_name_map,
                doc_indices=selected_docs,
                top_n_phrases_to_color=top_n,
                show_doc_separators=show_separators,
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Total phrases used in topic modelling", f"{len(core.phrase_occurrences)}", border=True)
            c2.metric("Sentences plotted (one dot per sentence)", f"{len(corpus_flow)}", border=True)
            c3.metric("Unique dominant phrases (anchors)", f"{corpus_flow["top_phrase"].nunique()}", border=True)

            with st.expander("💡 Understanding the Numbers", expanded=False):
                st.markdown(
                f"""
        ### Why are these three numbers different?
    
        **Simple answer:** This plot shows **one dot per sentence**, even when sentences contain multiple phrases.
    
        #### The Three Metrics Explained:
        
        #### 1) Total phrases used in topic modelling (e.g., {len(core.phrase_occurrences)})
        This is the size of your phrase vocabulary that went into clustering (unique phrases in `core_result.phrases_df`).
        
        - It answers: **“How many distinct phrases did PTM cluster?”**
        - It is **phrase-level**.
        
        #### 2) Sentences plotted (e.g., {len(corpus_flow)})
        This is the number of **key sentences** (sentences that contain ≥ 1 extracted phrase).
        
        - It answers: **“How many sentence locations contributed phrases?”**
        - It is **sentence-level**.
        - It can be lower or higher than phrase vocabulary depending on repetition.
        
        #### 3) Unique dominant phrases (e.g., {corpus_flow["top_phrase"].nunique()})
        Each sentence can contain multiple phrases, but the plot chooses exactly **one anchor phrase** per sentence
        (for coloring and legend). This is often based on **centrality** (or another rule).
        
        - It answers: **“Across all sentence dots, how many unique phrases actually became the anchor?”**
        - It is **sentence-anchoring**.
        - Many phrases will never become anchors because they appear inside sentences where another phrase
        wins the anchor rule.
        
        
        ---
    
        ### **Common Question: "Why does Cluster `C` look so small?"**
        
        **Answer:** Cluster `C`'s phrases might appear in many sentences, but those sentences are 
        *anchored* by phrases from other clusters (because those phrases are more central).
        
        **Solution:** Use the **Inspect View** tab to highlight cluster presence with the overlay feature!
        
        ---
        
        ### **How to Read This Plot:**
        
        1. **X-axis (timeline):** Sentence sequence through your corpus
        2. **Y-axis:** Dominant topic for each sentence
        3. **Color:** The most central phrase in that sentence
        4. **Gray dots:** Less important phrases (not in top N)
        
        **Pro tip:** Toggle legend items on/off to focus on specific phrases!
                    """
                )

            tab_default, tab_inspect = st.tabs(["Default View", "Inspect View"])

            with tab_default:
                st.info(
                    "**One dot per sentence.** Color = most central phrase in that sentence. "
                    "Gray = phrase not in top N. Toggle legend items to focus."
                )
                st.plotly_chart(corpus_flow_fig, use_container_width=True)
                download_plotly_as_html(corpus_flow_fig, "corpus_thematic_flow.html")

            with tab_inspect:
                st.warning(
                    "**Multi-cluster sentences explained:** A sentence with phrases from clusters `x`, `y`, and `z` appears as "
                    "ONE dot (colored by most central phrase). Use overlay below to see ALL clusters present in each sentence."
                )
                # Cluster presence overlay and table
                c1,c_ ,_= st.columns(3)
                focus_cluster_id = c1.number_input(
                    "Focus cluster ID", 
                    min_value=-1, value=0, step=1,
                    help="Overlay circles show ALL sentences containing this cluster's phrases"
                )
                if focus_cluster_id in corpus_flow['dominant_cluster'].values:
                    cluster_data = corpus_flow[corpus_flow['dominant_cluster'] == focus_cluster_id]
                    c1.metric(
                        f"Cluster {focus_cluster_id} Dominant", 
                        f"{len(cluster_data)} sentences",
                        help="Sentences where this cluster 'won' the anchor position",
                        border=True
                    )
                overlay_fig = add_cluster_presence_overlay(
                    timeline_result=timeline,
                    fig=corpus_flow_fig, 
                    corpus_flow=corpus_flow, 
                    focus_cluster_id=focus_cluster_id
                )
        
                st.plotly_chart(overlay_fig, use_container_width=True)

                # Enhanced data table showing cluster presence
                with st.expander(f"📋 All Sentences Containing Cluster {focus_cluster_id}", expanded=True):
                    # Get phrases from focus cluster
                    focus_phrases = core.phrases_df[
                        core.phrases_df['cluster_id'] == focus_cluster_id
                    ]['phrase'].tolist()
                    
                    # Filter sentences that contain any phrase from focus cluster
                    focus_sentences = corpus_flow[
                        corpus_flow['phrases'].apply(
                            lambda phrase_list: any(p in focus_phrases for p in phrase_list)
                        )
                    ].copy()
                    
                    if not focus_sentences.empty:
                        # Add cluster breakdown column
                        def get_cluster_breakdown(phrase_list):
                            clusters = []
                            for p in phrase_list:
                                cluster_info = core.phrases_df[core.phrases_df['phrase'] == p]
                                if not cluster_info.empty:
                                    cid = cluster_info.iloc[0]['cluster_id']
                                    clusters.append(f"{p} → Topic {cid}")
                            return clusters
                        
                        focus_sentences['all_phrases_with_clusters'] = focus_sentences['phrases'].apply(get_cluster_breakdown)
                        
                        # Show metrics
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Sentences", len(focus_sentences))
                        col2.metric("Where Dominant", (focus_sentences['dominant_cluster'] == focus_cluster_id).sum())
                        col3.metric("Where NOT Dominant", (focus_sentences['dominant_cluster'] != focus_cluster_id).sum())
                        
                        # Display table
                        st.dataframe(
                            focus_sentences[[
                                'doc_index', 
                                'sent_index', 
                                'dominant_cluster',
                                'top_phrase',
                                'all_phrases_with_clusters'
                            ]].rename(columns={
                                'all_phrases_with_clusters': 'All Phrases (with Topic IDs)'
                            }),
                            use_container_width=True,
                            height=400
                        )
                        
                        # Download option
                        csv = focus_sentences[[
                            'doc_index', 
                            'sent_index', 
                            'sentence_text',
                            'dominant_cluster',
                            'top_phrase',
                            'all_phrases_with_clusters'
                        ]].to_csv(index=False)
                        
                        st.download_button(
                            f"📥 Download Cluster {focus_cluster_id} Sentences",
                            csv,
                            f"cluster_{focus_cluster_id}_sentences.csv",
                            "text/csv"
                        )
                    else:
                        st.info(f"No sentences contain phrases from Cluster {focus_cluster_id}")
                        
        except Exception as e:
            st.error(f"Could not generate corpus flow: {e}")
    
    # ========================================================================
    # PHRASE IMPORTANCE ANALYSIS
    # ========================================================================
    
    with st.expander("📈 **Phrase Importance Analysis**", expanded=False):
        st.markdown("Analyze semantic importance combining frequency, length, and centrality.")
        
        try:
            phrase_importance_df = compute_phrase_importance(
                core_result=core,
                length_power=0.3,
                centrality_weight=0.9
            )
            
            st.dataframe(
                phrase_importance_df[[
                    "phrase", "count", "cluster_id", "num_words",
                    "freq_len_norm", "centrality_norm", "importance"
                ]],
                use_container_width=True,
                height=400
            )
            
            download_dataframe_as_csv(phrase_importance_df, "phrase_importance.csv")
            
        except Exception as e:
            st.error(f"Could not compute phrase importance: {e}")

    # ========================================================================
    # EXPLORE SUB-TOPICS
    # ========================================================================
    
    with st.expander("🔍 **Explore Sub-Topics**", expanded=False):
        st.markdown("Dive deep into a specific topic.")
        # Add cluster IDs to cluster names
        if labels and labels.cluster_name_map:
            cluster_name_map = {
                cid: f"{cid}. {name}" 
                for cid, name in labels.cluster_name_map.items()
            }
            
        options = sorted(core.phrases_df['cluster_id'].unique())
        
        def format_cluster(cid):
            if cid == -1:
                return "Cluster -1 (Noise)"
            elif cluster_name_map.get(cid):
                return f"Cluster {cid} | {cluster_name_map.get(cid)}"
            else:
                return f"Cluster {cid}"
        
        id_map = {format_cluster(cid): cid for cid in options}
        default_idx = 1 if len(id_map) > 1 and -1 in options else 0
        
        selected = st.selectbox('Select a topic:', list(id_map.keys()), index=default_idx)
        cid = id_map[selected]
        
        st.markdown(f"### {format_cluster(cid)}")
        
        if labels and cid in labels.labels_by_cluster:
            st.markdown(f"**Description:** {labels.labels_by_cluster[cid].description}")
        
        st.markdown("#### Sub-Topics")
        sub_cluster_df = sub_cluster_sentences(
            timeline_result=timeline,
            cluster_id=cid,
            embedding_model_name="all-mpnet-base-v2",
            hdbscan_min_cluster_size=3,
            hdbscan_min_samples=1
        )
        st.dataframe(sub_cluster_df, use_container_width=True)
        download_dataframe_as_csv(sub_cluster_df, f"cluster_{cid}_subtopics.csv")
    
    # ========================================================================
    # EXPORT & CODE GENERATION
    # ========================================================================
    
    with st.expander("📦 **Export Results & Code**", expanded=False):
        st.markdown("Download analysis results and get reproducible Python code.")
        
        st.markdown("### Python Code")
        try:
            code = generate_code_from_settings()
            st.code(code, language="python")
            
            st.download_button(
                label="📥 Download Python Script",
                data=code,
                file_name="ptm_analysis.py",
                mime="text/plain"
            )
        except Exception as e:
            st.warning(f"Could not generate code: {e}")
        
        st.markdown("### Data Exports")
        col1, col2 = st.columns(2)
        with col1:
            download_dataframe_as_csv(core.phrases_df, "phrases.csv")
        with col2:
            download_dataframe_as_csv(timeline.sentence_df, "timeline.csv")

if __name__ == "__main__":
    main()