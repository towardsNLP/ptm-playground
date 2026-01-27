"""
Results Section Helper Functions
Enhanced visualization and analysis for PhraseTopicMiner Playground
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import io

# ============================================================================
# SINGLE-CLUSTER VISUALIZATIONS
# ============================================================================

def create_cluster_bar_chart(core_result, cluster_id: int, top_n: int = 10) -> go.Figure:
    """
    Create horizontal bar chart showing top phrases in a cluster.
    
    Args:
        core_result: TopicCoreResult object
        cluster_id: Cluster ID
        top_n: Number of top phrases to show
        
    Returns:
        Plotly figure
    """
    # Get cluster phrases
    cluster_df = core_result.phrases_df[
        core_result.phrases_df['cluster_id'] == cluster_id
    ].copy()
    
    # Sort by count and take top N
    cluster_df = cluster_df.sort_values('count', ascending=False).head(top_n)
    
    # Reverse for plotting (highest at top)
    cluster_df = cluster_df.iloc[::-1]
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=cluster_df['count'],
        y=cluster_df['phrase'],
        orientation='h',
        marker=dict(
            color=cluster_df['count'],
            colorscale='Viridis',
            showscale=False
        ),
        text=cluster_df['count'],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Count: %{x}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Top {len(cluster_df)} Phrases",
        xaxis_title="Frequency",
        yaxis_title="",
        height=max(300, len(cluster_df) * 30),
        margin=dict(l=150, r=20, t=40, b=40),
        showlegend=False
    )
    
    return fig


def create_cluster_wordcloud(
    core_result, 
    cluster_id: int, 
    max_phrases: int = 50,
    width: int = 800,
    height: int = 400
) -> plt.Figure:
    """
    Create word cloud for a cluster, ensuring ALL requested phrases are visible.
    
    Args:
        core_result: TopicCoreResult object
        cluster_id: Cluster ID
        max_phrases: Maximum phrases to include
        width: Figure width in pixels
        height: Figure height in pixels
        
    Returns:
        Matplotlib figure
    """
    # Get cluster phrases
    cluster_df = core_result.phrases_df[
        core_result.phrases_df['cluster_id'] == cluster_id
    ].copy()
    
    # Sort by count and take top N
    cluster_df = cluster_df.sort_values('count', ascending=False).head(max_phrases)
    
    # Create frequency dict
    freq_dict = dict(zip(cluster_df['phrase'], cluster_df['count']))
    
    if not freq_dict:
        # Empty cluster - return empty figure
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.text(0.5, 0.5, 'No phrases in cluster', 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig
    
    # Create word cloud with optimized parameters
    wc = WordCloud(
        width=width,
        height=height,
        background_color='white',
        max_words=max_phrases,  # Ensure we try to fit all
        relative_scaling=0.5,  # Balance between frequency and fit
        min_font_size=8,  # Allow smaller fonts to fit more
        max_font_size=100,
        prefer_horizontal=0.7,
        colormap='viridis',
        collocations=False,  # Don't try to find phrases
        normalize_plurals=False
    )
    
    # Generate word cloud
    try:
        wc.generate_from_frequencies(freq_dict)
    except Exception as e:
        print(f"WordCloud generation failed: {e}")
        # Fallback: create simple text visualization
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        
        # Sort phrases by count
        sorted_phrases = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Display as text list
        y_pos = 0.9
        for phrase, count in sorted_phrases[:20]:  # Show top 20
            ax.text(0.1, y_pos, f"{phrase} ({count})", 
                   fontsize=10, va='top')
            y_pos -= 0.04
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(f"Top Phrases (WordCloud layout failed)")
        return fig
    
    # Create figure
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f"Cluster {cluster_id} Word Cloud ({len(freq_dict)} phrases)")
    
    plt.tight_layout(pad=0)
    
    return fig


# ============================================================================
# COMPREHENSIVE PHRASE DATAFRAME
# ============================================================================

def create_cluster_phrases_dataframe(
    core_result,
    cluster_id: int,
    timeline_result=None,
    sentences_by_doc: List[List[str]] = None,
    include_occurrences: bool = True
) -> pd.DataFrame:
    """
    Create comprehensive dataframe for all phrases in a cluster.
    
    Columns:
    - phrase, count, num_occurrences, kind, pattern, num_words
    - importance, freq_len_norm, centrality_norm
    - num_docs, doc_ids, x, y, centrality
    - occurrences (if include_occurrences=True)
    
    Args:
        core_result: TopicCoreResult object
        cluster_id: Cluster ID
        timeline_result: TimelineResult (for occurrences)
        sentences_by_doc: Nested list of sentences
        include_occurrences: Whether to include occurrence details
        
    Returns:
        DataFrame with comprehensive phrase information
    """
    # Start with phrases_df for this cluster
    cluster_df = core_result.phrases_df[
        core_result.phrases_df['cluster_id'] == cluster_id
    ].copy()
    
    # Sort by count descending
    cluster_df = cluster_df.sort_values('count', ascending=False).reset_index(drop=True)
    
    # Add kind and pattern from phrase_occurrences if available
    if hasattr(core_result, 'phrase_occurrences'):
        kinds = []
        patterns = []
        
        for phrase in cluster_df['phrase']:
            if phrase in core_result.phrase_occurrences:
                occs = core_result.phrase_occurrences[phrase]
                # Get kind and pattern from first occurrence
                if len(occs) > 0:
                    kinds.append(getattr(occs[0], 'kind', 'N/A'))
                    patterns.append(getattr(occs[0], 'pattern', 'N/A'))
                else:
                    kinds.append('N/A')
                    patterns.append('N/A')
            else:
                kinds.append('N/A')
                patterns.append('N/A')
        
        cluster_df['kind'] = kinds
        cluster_df['pattern'] = patterns
    
    # Add num_words if not present
    if 'num_words' not in cluster_df.columns:
        cluster_df['num_words'] = cluster_df['phrase'].str.split().str.len()
    
    # Compute importance metrics if not present
    if 'importance' not in cluster_df.columns:
        # Simple importance: frequency * length normalization
        max_count = cluster_df['count'].max()
        cluster_df['freq_len_norm'] = (
            (cluster_df['count'] / max_count) * 
            np.power(cluster_df['num_words'], 0.3)
        )
        
        # Centrality from x,y coordinates
        if 'x' in cluster_df.columns and 'y' in cluster_df.columns:
            # Distance from cluster centroid
            centroid_x = cluster_df['x'].mean()
            centroid_y = cluster_df['y'].mean()
            distances = np.sqrt(
                (cluster_df['x'] - centroid_x)**2 + 
                (cluster_df['y'] - centroid_y)**2
            )
            # Invert: closer to center = higher centrality
            max_dist = distances.max()
            cluster_df['centrality'] = distances
            cluster_df['centrality_norm'] = 1 - (distances / max_dist) if max_dist > 0 else 1.0
        else:
            cluster_df['centrality'] = 0.0
            cluster_df['centrality_norm'] = 0.0
        
        # Combined importance
        cluster_df['importance'] = (
            cluster_df['freq_len_norm'] * 0.6 + 
            cluster_df['centrality_norm'] * 0.4
        )
    
    # Add occurrence details if requested
    if include_occurrences and timeline_result is not None and sentences_by_doc is not None:
        num_occurrences_list = []
        doc_ids_list = []
        num_docs_list = []
        occurrences_list = []
        
        for phrase in cluster_df['phrase']:
            # Get occurrences from timeline or core result
            phrase_sents = timeline_result.phrase_sentence_df[
                timeline_result.phrase_sentence_df['phrase'] == phrase
            ]
            
            num_occ = len(phrase_sents)
            unique_docs = phrase_sents['doc_index'].unique().tolist()
            num_docs = len(unique_docs)
            
            # Format occurrences
            occurrences = []
            for _, row in phrase_sents.iterrows():
                doc_idx = row['doc_index']
                sent_idx = row['sent_index']
                sentence_text = sentences_by_doc[doc_idx][sent_idx]
                occurrences.append(f"(Doc {doc_idx}, Sent {sent_idx}): {sentence_text}")
            
            num_occurrences_list.append(num_occ)
            doc_ids_list.append(unique_docs)
            num_docs_list.append(num_docs)
            occurrences_list.append(occurrences) 
        
        cluster_df['num_occurrences'] = num_occurrences_list
        cluster_df['doc_ids'] = doc_ids_list
        cluster_df['num_docs'] = num_docs_list
        cluster_df['occurrences'] = occurrences_list
    
    # Reorder columns (proposed order)
    base_cols = ['phrase', 'count']
    
    if 'num_occurrences' in cluster_df.columns:
        base_cols.append('num_occurrences')
    
    base_cols.extend(['kind', 'pattern', 'num_words'])
    
    metric_cols = []
    if 'importance' in cluster_df.columns:
        metric_cols.extend(['importance', 'freq_len_norm', 'centrality_norm'])
    
    geo_cols = []
    if 'num_docs' in cluster_df.columns:
        geo_cols.extend(['num_docs', 'doc_ids'])
    if 'x' in cluster_df.columns:
        geo_cols.extend(['x', 'y', 'centrality'])
    
    occ_cols = []
    if 'occurrences' in cluster_df.columns:
        occ_cols.append('occurrences')
    
    # Combine in order
    ordered_cols = base_cols + metric_cols + geo_cols + occ_cols
    
    # Keep only columns that exist
    final_cols = [c for c in ordered_cols if c in cluster_df.columns]
    
    return cluster_df[final_cols]


# ============================================================================
# OCCURRENCE VIEWER
# ============================================================================

def get_phrase_occurrences(
    phrase: str,
    core_result,
    timeline_result,
    sentences_by_doc: List[List[str]]
) -> List[str]:
    """
    Get all formatted occurrences for a specific phrase.
    
    Args:
        phrase: Phrase text
        core_result: TopicCoreResult
        timeline_result: TimelineResult
        sentences_by_doc: Nested list of sentences
        
    Returns:
        List of formatted occurrences: "(Doc X, Sent Y): sentence text"
    """
    occurrences = []
    
    # Get from timeline
    phrase_sents = timeline_result.phrase_sentence_df[
        timeline_result.phrase_sentence_df['phrase'] == phrase
    ]
    
    for _, row in phrase_sents.iterrows():
        doc_idx = row['doc_index']
        sent_idx = row['sent_index']
        
        try:
            sentence_text = sentences_by_doc[doc_idx][sent_idx]
            occ = f"(Doc {doc_idx}, Sent {sent_idx}): {sentence_text}"
            occurrences.append(occ)
        except (IndexError, KeyError):
            # Sentence not found
            occurrences.append(f"(Doc {doc_idx}, Sent {sent_idx}): [Sentence not found]")
    
    return occurrences


# ============================================================================
# CLUSTER SENTENCES DATAFRAME
# ============================================================================

def create_cluster_sentences_dataframe(
    core_result,
    timeline_result,
    cluster_id: int,
    sentences_by_doc: List[List[str]]
) -> pd.DataFrame:
    """
    Create dataframe of all sentences in a cluster.
    """
    # Get phrase dataframe for this cluster (has centrality)
    phrase_df = create_cluster_phrases_dataframe(
        core_result=core_result,
        cluster_id=cluster_id,
        timeline_result=timeline_result,
        sentences_by_doc=sentences_by_doc,
        include_occurrences=False
    )
    
    if phrase_df.empty:
        return pd.DataFrame()
    
    # Create lookups from phrase_df
    phrase_to_centrality = dict(zip(phrase_df['phrase'], phrase_df['centrality']))
    phrase_to_count = dict(zip(phrase_df['phrase'], phrase_df['count']))
    
    # Get list of phrases in this cluster
    cluster_phrases = phrase_df['phrase'].tolist()
    
    # Build dict: (doc_index, sent_index) -> [phrases]
    sentence_to_phrases = {}
    
    for phrase in cluster_phrases:
        if phrase in core_result.phrase_occurrences:
            for occurrence in core_result.phrase_occurrences[phrase]:
                key = (occurrence.doc_index, occurrence.sent_index)
                if key not in sentence_to_phrases:
                    sentence_to_phrases[key] = []
                sentence_to_phrases[key].append(phrase)
    
    if not sentence_to_phrases:
        return pd.DataFrame()
    
    # Build rows
    rows = []
    for (doc_idx, sent_idx), phrases in sentence_to_phrases.items():
        # Get sentence text
        try:
            sentence_text = sentences_by_doc[doc_idx][sent_idx]
        except (IndexError, KeyError):
            sentence_text = "[Sentence not found]"
        
        # Get centrality for each phrase
        centralities = [phrase_to_centrality.get(p, 0.0) for p in phrases]
        counts = [phrase_to_count.get(p, 0) for p in phrases]
        
        rows.append({
            'doc_index': doc_idx,
            'sent_index': sent_idx,
            'sentence_text': sentence_text,
            'phrases': phrases, #
            'num_phrases': len(phrases),
            'max_centrality': min(centralities) if centralities else None,
            'avg_centrality': np.mean(centralities) if centralities else None,
            'phrase_counts': [f"{p}({c})" for p, c in zip(phrases, counts)]
        })
    
    # Create and sort dataframe
    df = pd.DataFrame(rows)
    df = df.sort_values(['doc_index', 'sent_index']).reset_index(drop=True)
    
    return df











