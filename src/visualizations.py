"""
Visualization Functions for PTM Playground

This module contains all visualization functions including:
- Thematic flow plots
- Topic summary grids (bar charts)
- Word cloud grids
- Interactive DataMapPlot visualizations

All visualizations support download as images or HTML.
"""
import streamlit as st
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib import cm
from wordcloud import WordCloud
from PIL import Image
from typing import Dict, List, Optional, Tuple, Callable
import os

import html, re, textwrap

from phrasetopicminer import TopicCoreResult, TopicTimelineResult, TopicLabelingResult
from .utils import highlight_phrases_in_sentence, build_hover_html, get_phrase_count


def plot_corpus_thematic_flow(
    doc_topic_profile: pd.DataFrame,
    timeline_result: TopicTimelineResult,
    core_result: TopicCoreResult,
    cluster_name_map: Optional[Dict[int, str]] = None,
    doc_indices: Optional[List[int]] = None,
    top_n_phrases_to_color: int = 30,
    show_doc_separators: bool = False
) -> go.Figure:
    """
    Visualizes the thematic flow across the corpus or a subset of documents.

    This powerful, consolidated function can operate on the entire corpus or be filtered
    to specific documents. It reveals how topics evolve throughout the text.

    Points are colored by their most semantically important phrase. Only the top N
    phrases receive a unique color for visual clarity, while the rest are grey.
    The legend is complete, allowing users to toggle any phrase on or off.

    Args:
        doc_topic_profile: Output from create_document_topic_profile
        timeline_result: For accessing sentence data
        core_result: For calculating phrase importance and counts
        cluster_name_map: Dict mapping cluster_id to label string (optional)
        doc_indices: A list of doc IDs. If None, all docs are shown
        top_n_phrases_to_color: Number of top phrases to assign a unique color
        show_doc_separators: Draws vertical lines at document boundaries

    Returns:
        A Plotly Graph Objects Figure
    """
    
    # 1. Prepare the base dataframe
    corpus_flow = doc_topic_profile.reset_index().merge(
        timeline_result.sentence_df[['doc_index', 'sent_index', 'timeline_idx', 'sentence_text']],
        on=['doc_index', 'sent_index']
    )
    phrases_in_sent = timeline_result.phrase_sentence_df.groupby(['doc_index', 'sent_index'])['phrase'].apply(list).rename('phrases')
    corpus_flow = corpus_flow.merge(phrases_in_sent, on=['doc_index', 'sent_index'], how='left')
    corpus_flow['phrases'] = corpus_flow['phrases'].apply(lambda v: v if isinstance(v, list) else [])

    # 2. Filter by selected documents
    if doc_indices is not None and len(doc_indices) > 0:
        corpus_flow = corpus_flow[corpus_flow['doc_index'].isin(doc_indices)].copy()
        if corpus_flow.empty:
            return go.Figure().update_layout(title="No data for selected documents.")

    # 3. Use centrality to determine dominant phrase
    phrase_info_df = core_result.phrases_df[['phrase', 'count']].copy()
    
    # Get or compute centrality
    if 'centrality' in core_result.phrases_df.columns:
        phrase_info_df['centrality'] = core_result.phrases_df['centrality']
    elif 'x' in core_result.phrases_df.columns and 'y' in core_result.phrases_df.columns:
        centroid_x = core_result.phrases_df['x'].mean()
        centroid_y = core_result.phrases_df['y'].mean()
        distances = np.sqrt(
            (core_result.phrases_df['x'] - centroid_x)**2 + 
            (core_result.phrases_df['y'] - centroid_y)**2
        )
        phrase_info_df['centrality'] = distances
    else:
        phrase_info_df['centrality'] = 0
    
    phrase_info_lookup = phrase_info_df.set_index('phrase')
    
    def get_most_central_phrase(phrases):
        if not phrases:
            return None
        # Lower centrality = more central = better
        return min(phrases, key=lambda p: phrase_info_lookup.loc[p, 'centrality'] if p in phrase_info_lookup.index else float('inf'))
    
    corpus_flow['top_phrase'] = corpus_flow['phrases'].apply(get_most_central_phrase)

    # 4. Create legend labels that include the frequency count
    # Store FULL label
    corpus_flow['legend_label_full'] = corpus_flow['top_phrase'].apply(
        lambda p: f"{p} (C: {get_phrase_count(p, phrase_info_df)})"
    )
    
    # Create TRUNCATED label for legend (max 50 chars)
    corpus_flow['legend_label'] = corpus_flow['legend_label_full'].apply(
        lambda label: label[:50] + '...' if len(label) > 50 else label
    )

    # 5. Color the most central phrases
    top_n_phrases = phrase_info_lookup['centrality'].nsmallest(top_n_phrases_to_color).index.tolist()
    
    # Get all unique legend labels and sort by centrality (ascending = most central first)
    unique_labels_df = corpus_flow[['top_phrase', 'legend_label']].drop_duplicates()
    unique_labels_df['centrality'] = unique_labels_df['top_phrase'].map(phrase_info_lookup['centrality'])
    sorted_labels = unique_labels_df.sort_values('centrality', ascending=True)['legend_label'].tolist()  # FIXED: ascending=True
    
    # Create the color map
    color_map = {}
    color_palette = px.colors.qualitative.Plotly
    color_idx = 0
    for label in sorted_labels:
        # Get phrase from the unique_labels_df to avoid truncation issues
        phrase = unique_labels_df[unique_labels_df['legend_label'] == label]['top_phrase'].iloc[0]
        if phrase in top_n_phrases:
            color_map[label] = color_palette[color_idx % len(color_palette)]
            color_idx += 1
        else:
            color_map[label] = 'lightgrey'  # FIXED: lightgrey instead of white

    # 6. Prepare hover text and other labels
    corpus_flow['highlighted_sentence'] = corpus_flow.apply(
        lambda row: highlight_phrases_in_sentence(row['sentence_text'], row['phrases']), 
        axis=1
    )
    y_axis_label = 'dominant_cluster'
    has_labels = cluster_name_map is not None and len(cluster_name_map) > 0
    if has_labels:
        # Store FULL label first
        corpus_flow['topic_label_full'] = corpus_flow['dominant_cluster'].apply(
            lambda cid: cluster_name_map.get(cid, f"Cluster {cid}")
        )
        # Then create TRUNCATED version for display
        corpus_flow['topic_label'] = corpus_flow['topic_label_full'].apply(
            lambda label: label[:40] + '...' if len(label) > 40 else label
        )
        y_axis_label = 'topic_label'
        
    corpus_flow['hover_html'] = corpus_flow.apply(
        lambda row: build_hover_html(row, has_labels), 
        axis=1
    )

    # 7. Create the plot
    fig = px.scatter(
        corpus_flow.sort_values('timeline_idx'),
        x='timeline_idx',
        y=y_axis_label,
        color='legend_label',
        color_discrete_map=color_map,
        category_orders={"legend_label": sorted_labels},
        title=("Corpus-Wide Thematic Flow" if not doc_indices else f"Thematic Flow for Doc(s): {doc_indices}"),
        labels={
            'timeline_idx': 'Global Key Sentence Sequence',
            y_axis_label: 'Dominant Topic',
            'legend_label': 'Dominant Phrase'
        },
        custom_data=['hover_html'],
    )
    
    # Update hovertemplate and marker style
    fig.update_traces(
        marker=dict(size=8, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')),
        hovertemplate="%{customdata[0]}<extra></extra>",
    )

    # Add document separators
    if show_doc_separators and (not doc_indices or len(doc_indices) > 1):
        doc_boundaries = corpus_flow.groupby('doc_index')['timeline_idx'].max().sort_index()
        for i in range(len(doc_boundaries) - 1):
            boundary_pos = (
                doc_boundaries.iloc[i]
                + corpus_flow[corpus_flow['doc_index'] == doc_boundaries.index[i + 1]]['timeline_idx'].min()
            ) / 2
            fig.add_vline(
                x=boundary_pos,
                line_width=1,
                line_dash="dash",
                line_color="grey",
                annotation_text=f"Doc {doc_boundaries.index[i + 1]} →",
                annotation_position="top left",
            )

    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        hoverlabel=dict(bgcolor="white", font_size=13),
        width=None,
        height=800,
        margin=dict(l=100)  # ADD THIS - increase left margin
    )

    return fig, corpus_flow


def add_cluster_presence_overlay(
    timeline_result, 
    fig: go.Figure, 
    corpus_flow: pd.DataFrame, 
    focus_cluster_id: int
) -> go.Figure:
    """
    Overlay a highlight trace for sentences that *contain* a given cluster_id.
    This keeps the base plot "one point per sentence" while making multi-cluster
    membership visible via an outline layer.
    """
    # cluster membership per sentence
    sent_clusters = (
        corpus_flow[["doc_index", "sent_index"]]
        .merge(
            timeline_result.phrase_sentence_df[["doc_index", "sent_index", "cluster_id"]],
            on=["doc_index", "sent_index"],
            how="left",
        )
    )

    has_focus = (
        sent_clusters.groupby(["doc_index", "sent_index"])["cluster_id"]
        .apply(lambda s: focus_cluster_id in set(s.dropna().astype(int)))
        .rename("has_focus")
        .reset_index()
    )

    overlay_df = corpus_flow.merge(has_focus, on=["doc_index", "sent_index"], how="left")
    overlay_df = overlay_df[overlay_df["has_focus"] == True]

    if overlay_df.empty:
        return fig

    fig.add_trace(
        go.Scatter(
            x=overlay_df["timeline_idx"],
            y=overlay_df["topic_label"] if "topic_label" in overlay_df.columns else overlay_df["dominant_cluster"],
            mode="markers",
            name=f"Contains Cluster {focus_cluster_id}",
            marker=dict(
                size=12,
                symbol="circle-open",
                line=dict(width=2),
            ),
            hoverinfo="skip",  # base trace already has rich hover
            showlegend=True,
        )
    )
    return fig


# # from __future__ import annotations

# # from dataclasses import dataclass
# # from typing import Dict, List, Optional, Tuple

# # import html
# # import numpy as np
# # import pandas as pd
# # import plotly.express as px
# # import plotly.graph_objects as go


# def plot_corpus_thematic_flow(
#     doc_topic_profile: Optional[pd.DataFrame],  # kept for backward compatibility; not used
#     timeline_result,
#     core_result,
#     cluster_name_map: Optional[Dict[int, str]] = None,
#     doc_indices: Optional[List[int]] = None,
#     top_n_phrases_to_color: int = 30,
#     show_doc_separators: bool = False,
#     *,
#     noise_cluster_id: int = -1,
#     prefer_non_noise_anchor: bool = True,
#     truncate_cluster_labels: int = 50,
#     truncate_phrase_labels: int = 60,
# ) -> Tuple[go.Figure, pd.DataFrame]:
#     """
#     Plot a corpus-wide “thematic flow” with **one point per sentence**.

#     Concept
#     -------
#     - Each point = one sentence that contains ≥1 clustered phrase.
#     - The sentence is assigned an **anchor phrase** (for color) and an **anchor cluster**
#       (for the y-axis row).
#     - If a sentence contains phrases from multiple clusters, only the anchor cluster
#       determines the y position. Use overlays (see `add_cluster_presence_overlay`) to
#       reveal multi-cluster membership without duplicating points.

#     Data sources
#     ------------
#     This function intentionally uses `timeline_result.phrase_sentence_df` as the
#     canonical input. That table already exists specifically to map phrase occurrences
#     back to sentence context (doc/sent indices, sentence text, and timeline order).  
#     It is built from `core_result.phrase_occurrences` plus `core_result.phrases_df`
#     lookups. (So you don’t need to re-walk `phrase_occurrences` here.)

#     Anchor phrase selection
#     -----------------------
#     We compute a centrality score per phrase as its distance to the centroid of its
#     *own cluster* (prefer embeddings, fall back to x/y). Lower = more central.

#     For each sentence:
#     1) Rank its phrases by:
#        - (optional) prefer non-noise phrases over noise (`cluster_id != -1`)
#        - centrality ascending (closest to centroid)
#        - count descending (more frequent phrase wins ties)
#        - n_tokens descending (longer phrase wins final ties)
#     2) Pick the best as `anchor_phrase`.
#     3) Set `anchor_cluster_id = cluster_id(anchor_phrase)`.

#     Noise handling (UI heuristic)
#     -----------------------------
#     If `prefer_non_noise_anchor=True` and the most central phrase is noise, we pick the
#     best non-noise phrase in that sentence. Noise is only used if the sentence contains
#     only noise phrases.

#     Parameters
#     ----------
#     doc_topic_profile:
#         Deprecated for this plot. Kept only so you can drop-in replace your existing call
#         sites. You can pass None safely.
#     timeline_result:
#         Must expose `.phrase_sentence_df` with columns including:
#         phrase, cluster_id, doc_index, sent_index, sentence_text, timeline_idx, ...
#     core_result:
#         Must expose `.phrases_df` with at least phrase/count/cluster_id and ideally
#         embedding (or x,y for fallback).
#     cluster_name_map:
#         Optional mapping cluster_id -> human label (e.g., from TopicLabeler).
#     doc_indices:
#         Optional list of doc_index values to include. None means “all docs”.
#     top_n_phrases_to_color:
#         Assign distinct colors only to the top N anchor phrases (ranked by centrality).
#         All other anchor phrases are rendered lightgrey to avoid a rainbow legend.
#     show_doc_separators:
#         Draw vertical separators between documents (based on timeline_idx).
#     noise_cluster_id:
#         Cluster id used for noise. (HDBSCAN convention is -1.)
#     prefer_non_noise_anchor:
#         If True, avoid choosing a noise phrase as the anchor when non-noise phrases exist.
#     truncate_cluster_labels / truncate_phrase_labels:
#         Purely cosmetic truncation for display/legend names.

#     Returns
#     -------
#     (fig, corpus_flow):
#         fig:
#             Plotly figure.
#         corpus_flow:
#             One row per plotted sentence, including:
#             doc_index, sent_index, timeline_idx, sentence_text, phrases,
#             clusters_present, anchor_phrase, anchor_cluster_id, anchor_cluster_label, etc.
#     """

#     # ------------------------------------------------------------------
#     # 1) Start from phrase_sentence_df and filter docs (if requested)
#     # ------------------------------------------------------------------
#     psdf = timeline_result.phrase_sentence_df.copy()
#     if psdf.empty:
#         return go.Figure().update_layout(title="No phrase-sentence data."), pd.DataFrame()

#     if doc_indices:
#         psdf = psdf[psdf["doc_index"].isin(doc_indices)].copy()
#         if psdf.empty:
#             return go.Figure().update_layout(title="No data for selected documents."), pd.DataFrame()

#     # ------------------------------------------------------------------
#     # 2) Aggregate to ONE ROW PER SENTENCE:
#     #    - phrases: list of phrases in that sentence (deduped, stable order)
#     #    - clusters_present: set of cluster_ids present in that sentence
#     #    - keep timeline_idx + sentence_text from first occurrence
#     # ------------------------------------------------------------------
#     def _unique_stable(seq: List[str]) -> List[str]:
#         seen = set()
#         out = []
#         for x in seq:
#             if x not in seen:
#                 seen.add(x)
#                 out.append(x)
#         return out

#     sent_agg = (
#         psdf.sort_values(["timeline_idx", "occurrence_index"], ascending=True)
#         .groupby(["doc_index", "sent_index"], as_index=False)
#         .agg(
#             timeline_idx=("timeline_idx", "first"),
#             sentence_text=("sentence_text", "first"),
#             phrases=("phrase", lambda s: _unique_stable(list(s))),
#             clusters_present=("cluster_id", lambda s: set(int(v) for v in pd.Series(s).dropna().tolist())),
#         )
#     )

#     # ------------------------------------------------------------------
#     # 3) Build phrase lookup table from core_result.phrases_df
#     # ------------------------------------------------------------------
#     pdf = core_result.phrases_df.copy()
#     needed = {"phrase", "count", "cluster_id"}
#     missing = needed - set(pdf.columns)
#     if missing:
#         raise ValueError(f"core_result.phrases_df is missing required columns: {missing}")

#     # Optional columns for better tie-breaking / centrality
#     if "n_tokens" not in pdf.columns:
#         pdf["n_tokens"] = pdf["phrase"].astype(str).apply(lambda p: len(p.split()))

#     phrase_lookup = pdf.set_index("phrase")[["count", "cluster_id", "n_tokens"]].copy()

#     # ------------------------------------------------------------------
#     # 4) Compute within-cluster centrality: distance to that cluster’s centroid
#     #    Prefer embeddings, fall back to x/y, else constant 0.
#     # ------------------------------------------------------------------
#     centrality = pd.Series(index=pdf["phrase"], dtype=float)

#     has_embeddings = "embedding" in pdf.columns and pdf["embedding"].notna().any()
#     has_xy = "x" in pdf.columns and "y" in pdf.columns and pdf[["x", "y"]].notna().all(axis=1).any()

#     if has_embeddings:
#         # Compute centroid per cluster in embedding space
#         # (embeddings stored as np.ndarray objects per TopicCoreResult docstring)
#         centroids: Dict[int, np.ndarray] = {}
#         for cid, sub in pdf.groupby("cluster_id"):
#             vecs = [v for v in sub["embedding"].tolist() if isinstance(v, np.ndarray)]
#             if not vecs:
#                 continue
#             centroids[int(cid)] = np.mean(np.stack(vecs, axis=0), axis=0)

#         vals = []
#         for row in pdf.itertuples(index=False):
#             phrase = row.phrase
#             cid = int(row.cluster_id)
#             emb = row.embedding
#             c = centroids.get(cid)
#             if isinstance(emb, np.ndarray) and c is not None:
#                 vals.append((phrase, float(np.linalg.norm(emb - c))))
#             else:
#                 vals.append((phrase, float("inf")))
#         centrality = pd.Series(dict(vals), dtype=float)

#     elif has_xy:
#         # Compute centroid per cluster in x/y space
#         centroids_xy = (
#             pdf.groupby("cluster_id")[["x", "y"]].mean().rename(columns={"x": "cx", "y": "cy"})
#         )
#         vals = []
#         for row in pdf.itertuples(index=False):
#             phrase = row.phrase
#             cid = int(row.cluster_id)
#             try:
#                 cx, cy = float(centroids_xy.loc[cid, "cx"]), float(centroids_xy.loc[cid, "cy"])
#                 d = float(np.sqrt((row.x - cx) ** 2 + (row.y - cy) ** 2))
#                 vals.append((phrase, d))
#             except Exception:
#                 vals.append((phrase, float("inf")))
#         centrality = pd.Series(dict(vals), dtype=float)

#     else:
#         centrality = pd.Series({p: 0.0 for p in pdf["phrase"].tolist()}, dtype=float)

#     phrase_lookup["centrality"] = centrality
#     # Make sure every phrase has an entry (some sentences might contain filtered phrases)
#     phrase_lookup["centrality"] = phrase_lookup["centrality"].fillna(float("inf"))

#     # ------------------------------------------------------------------
#     # 5) Pick anchor phrase per sentence (optionally skip noise)
#     # ------------------------------------------------------------------
#     def _rank_key(p: str) -> Tuple[int, float, int, int]:
#         """
#         Sorting key:
#         - noise penalty (0 for non-noise, 1 for noise) if prefer_non_noise_anchor
#         - centrality ascending
#         - count descending
#         - n_tokens descending
#         """
#         if p not in phrase_lookup.index:
#             # Unknown phrase: push to bottom
#             return (1, float("inf"), -1, -1)

#         row = phrase_lookup.loc[p]
#         cid = int(row["cluster_id"])
#         is_noise = int(cid == noise_cluster_id) if prefer_non_noise_anchor else 0
#         cent = float(row["centrality"])
#         cnt = int(row["count"])
#         nt = int(row["n_tokens"])
#         return (is_noise, cent, -cnt, -nt)

#     def choose_anchor_phrase(phrases: List[str]) -> Optional[str]:
#         if not phrases:
#             return None
#         # Choose the best phrase by our ranking
#         best = min(phrases, key=_rank_key)

#         # If best is noise but non-noise exists, min() above already avoids noise
#         # (because noise gets a penalty). So we can just return best.
#         return best

#     sent_agg["anchor_phrase"] = sent_agg["phrases"].apply(choose_anchor_phrase)
#     sent_agg = sent_agg.dropna(subset=["anchor_phrase"]).copy()

#     sent_agg["anchor_cluster_id"] = sent_agg["anchor_phrase"].apply(
#         lambda p: int(phrase_lookup.loc[p, "cluster_id"]) if p in phrase_lookup.index else noise_cluster_id
#     )

#     # Optional human-readable cluster labels
#     if cluster_name_map:
#         def _clabel(cid: int) -> str:
#             label = cluster_name_map.get(cid, f"Cluster {cid}")
#             label = str(label)
#             return label if len(label) <= truncate_cluster_labels else (label[:truncate_cluster_labels] + "…")

#         sent_agg["anchor_cluster_label"] = sent_agg["anchor_cluster_id"].apply(_clabel)
#         y_col = "anchor_cluster_label"
#     else:
#         sent_agg["anchor_cluster_label"] = sent_agg["anchor_cluster_id"].astype(str)
#         y_col = "anchor_cluster_label"

#     # ------------------------------------------------------------------
#     # 6) Colors: only top-N anchor phrases get unique colors, rest lightgrey
#     # ------------------------------------------------------------------
#     anchor_phrases = sent_agg["anchor_phrase"].unique().tolist()
#     anchor_info = phrase_lookup.loc[phrase_lookup.index.intersection(anchor_phrases)].copy()

#     # Rank “top phrases” by centrality (smaller = more central)
#     top_phrases = (
#         anchor_info.sort_values("centrality", ascending=True)
#         .head(max(int(top_n_phrases_to_color), 0))  # allow 0
#         .index
#         .tolist()
#     )

#     palette = px.colors.qualitative.Plotly
#     color_map: Dict[str, str] = {}
#     color_i = 0
#     for p in anchor_phrases:
#         if p in top_phrases:
#             color_map[p] = palette[color_i % len(palette)]
#             color_i += 1
#         else:
#             color_map[p] = "lightgrey"

#     # Create a nicer legend name per phrase (without using it as a key)
#     def _phrase_display(p: str) -> str:
#         if p not in phrase_lookup.index:
#             return str(p)
#         cnt = int(phrase_lookup.loc[p, "count"])
#         text = f"{p} (C: {cnt})"
#         return text if len(text) <= truncate_phrase_labels else (text[:truncate_phrase_labels] + "…")

#     phrase_display_map = {p: _phrase_display(p) for p in anchor_phrases}

#     # Sort legend categories by centrality for readability
#     order = (
#         anchor_info.reset_index()
#         .sort_values("centrality", ascending=True)["phrase"]
#         .tolist()
#     )

#     # ------------------------------------------------------------------
#     # 7) Hover HTML (simple + readable). Customize as you like.
#     # ------------------------------------------------------------------
#     def _escape(s: str) -> str:
#         return html.escape(str(s))

#     def build_hover(row) -> str:
#         p = row["anchor_phrase"]
#         cid = int(row["anchor_cluster_id"])
#         sent = row["sentence_text"]
#         phrase_list = row["phrases"]

#         phrases_html = ", ".join(_escape(x) for x in phrase_list[:30])
#         if len(phrase_list) > 30:
#             phrases_html += f" … (+{len(phrase_list)-30} more)"

#         return (
#             f"<b>Doc</b>: {row['doc_index']} &nbsp; <b>Sent</b>: {row['sent_index']}<br>"
#             f"<b>TimeIdx</b>: {row['timeline_idx']}<br>"
#             f"<b>Anchor Cluster</b>: {_escape(row['anchor_cluster_label'])} (id={cid})<br>"
#             f"<b>Anchor Phrase</b>: {_escape(p)}<br>"
#             f"<b>All phrases in sentence</b>: {phrases_html}<br><br>"
#             f"<b>Sentence</b>: {_escape(sent)}"
#         )

#     sent_agg["hover_html"] = sent_agg.apply(build_hover, axis=1)

#     corpus_flow = sent_agg  # naming for your downstream overlay code

#     # ------------------------------------------------------------------
#     # 8) Build the plot
#     # ------------------------------------------------------------------
#     fig = px.scatter(
#         corpus_flow.sort_values("timeline_idx"),
#         x="timeline_idx",
#         y=y_col,
#         color="anchor_phrase",                 # KEY: raw phrase as category key
#         color_discrete_map=color_map,
#         category_orders={"anchor_phrase": order},
#         labels={
#             "timeline_idx": "Global Key Sentence Sequence",
#             y_col: "Anchor Topic",
#             "anchor_phrase": "Anchor Phrase",
#         },
#         custom_data=["hover_html"],
#         title=("Corpus-Wide Thematic Flow" if not doc_indices else f"Thematic Flow for Doc(s): {doc_indices}"),
#     )

#     # Replace trace names with truncated display labels (without breaking color keys)
#     for tr in fig.data:
#         raw = tr.name
#         if raw in phrase_display_map:
#             tr.name = phrase_display_map[raw]

#     fig.update_traces(
#         marker=dict(size=8, opacity=0.85, line=dict(width=1, color="DarkSlateGrey")),
#         hovertemplate="%{customdata[0]}<extra></extra>",
#     )

#     # Optional document separators
#     if show_doc_separators:
#         doc_boundaries = corpus_flow.groupby("doc_index")["timeline_idx"].max().sort_index()
#         for i in range(len(doc_boundaries) - 1):
#             left_doc = doc_boundaries.index[i]
#             right_doc = doc_boundaries.index[i + 1]

#             left_max = float(doc_boundaries.iloc[i])
#             right_min = float(corpus_flow.loc[corpus_flow["doc_index"] == right_doc, "timeline_idx"].min())
#             boundary_pos = (left_max + right_min) / 2.0

#             fig.add_vline(
#                 x=boundary_pos,
#                 line_width=1,
#                 line_dash="dash",
#                 line_color="grey",
#                 annotation_text=f"Doc {right_doc} →",
#                 annotation_position="top left",
#             )

#     fig.update_layout(
#         height=800,
#         margin=dict(l=120, r=20, t=60, b=40),
#         hoverlabel=dict(bgcolor="white", font_size=13),
#     )

#     return fig, corpus_flow


# def add_cluster_presence_overlay(
#     fig: go.Figure,
#     corpus_flow: pd.DataFrame,
#     focus_cluster_id: int,
#     *,
#     y_col: str = "anchor_cluster_label",
#     name: Optional[str] = None,
# ) -> go.Figure:
#     """
#     Overlay a highlight ring on sentences that *contain* a given cluster_id,
#     without duplicating points.

#     This expects `corpus_flow` produced by `plot_corpus_thematic_flow`, which includes:
#     - timeline_idx
#     - clusters_present (a set of cluster_ids per sentence)
#     - anchor_cluster_label (or your chosen y_col)
#     """
#     if corpus_flow.empty or "clusters_present" not in corpus_flow.columns:
#         return fig

#     mask = corpus_flow["clusters_present"].apply(lambda s: focus_cluster_id in (s or set()))
#     overlay_df = corpus_flow[mask].copy()
#     if overlay_df.empty:
#         return fig

#     fig.add_trace(
#         go.Scatter(
#             x=overlay_df["timeline_idx"],
#             y=overlay_df[y_col],
#             mode="markers",
#             name=name or f"Contains Cluster {focus_cluster_id}",
#             marker=dict(size=12, symbol="circle-open", line=dict(width=2)),
#             hoverinfo="skip",
#             showlegend=True,
#         )
#     )
#     return fig


# def plot_corpus_thematic_flow(
#     doc_topic_profile: pd.DataFrame,
#     timeline_result,
#     core_result,
#     cluster_name_map: Optional[Dict[int, str]] = None,
#     doc_indices: Optional[List[int]] = None,
#     top_n_phrases_to_color: int = 30,
#     show_doc_separators: bool = False,
#     use_phrase_occurrences: bool = True,
#     legend_max_chars: int = 60,
# ) -> go.Figure:
#     """
#     Plot the corpus thematic flow with **one point per sentence that contains ≥1 phrase**.

#     Design goals
#     ------------
#     - Include ALL sentences that contain phrases (based on phrase occurrences).
#     - Each sentence appears exactly once, even if it contains many phrases.
#     - Color points by a single “dominant phrase” (here: most-central among phrases in the sentence).
#     - Keep legend “complete” for the phrases that actually become dominant phrases.
#     - Avoid Plotly category collisions by NEVER using truncated text as the grouping key.

#     Why your prior version dropped phrases
#     --------------------------------------
#     You truncated legend labels (e.g., first 50 chars) and then used that truncated string
#     as the `color` category. Different phrases can share the same prefix → category collision.
#     Your `drop_duplicates()` on the truncated label then permanently removed some phrases.

#     Parameters
#     ----------
#     doc_topic_profile:
#         DataFrame indexed by ['doc_index','sent_index'] (or containing these columns)
#         with a 'dominant_cluster' column.
#     timeline_result:
#         Must have sentence_df with columns: ['doc_index','sent_index','timeline_idx','sentence_text'].
#     core_result:
#         Must have phrases_df with columns ['phrase','count'] and ideally 'centrality' or ('x','y').
#         Also may have phrase_occurrences: dict[str, list[PhraseRecord]].
#     cluster_name_map:
#         Optional mapping cluster_id -> label. If given, y-axis uses labels.
#     doc_indices:
#         Optional list of docs to include.
#     top_n_phrases_to_color:
#         Only the top-N most central phrases get a distinct color; others are lightgrey.
#         (They still appear as separate legend items if they are dominant phrases somewhere.)
#     show_doc_separators:
#         Draw vertical lines between documents.
#     use_phrase_occurrences:
#         If True, build the sentence→phrases mapping from core_result.phrase_occurrences
#         (more robust / direct). If False, use timeline_result.phrase_sentence_df.
#     legend_max_chars:
#         Max chars for legend display names (display only; grouping uses full keys).

#     Returns
#     -------
#     plotly.graph_objects.Figure
#     """

#     # -----------------------------
#     # Helpers
#     # -----------------------------
#     def _truncate(s: str, n: int) -> str:
#         s = "" if s is None else str(s)
#         return s if len(s) <= n else (s[: n - 1] + "…")

#     def _make_unique(names: List[str]) -> List[str]:
#         """Make duplicate display names unique by appending (2), (3), ..."""
#         seen: Dict[str, int] = {}
#         out: List[str] = []
#         for name in names:
#             k = name
#             seen[k] = seen.get(k, 0) + 1
#             out.append(k if seen[k] == 1 else f"{k} ({seen[k]})")
#         return out

#     # -----------------------------
#     # 1) Build sentence -> phrases (one row per sentence)
#     # -----------------------------
#     if use_phrase_occurrences and hasattr(core_result, "phrase_occurrences") and core_result.phrase_occurrences:
#         # phrase_occurrences: dict[str, list[PhraseRecord(doc_index, sent_index, ...)]]
#         rows: List[Tuple[int, int, str]] = []
#         for phrase, recs in core_result.phrase_occurrences.items():
#             for r in recs:
#                 rows.append((int(r.doc_index), int(r.sent_index), str(phrase)))

#         occ_df = pd.DataFrame(rows, columns=["doc_index", "sent_index", "phrase"])
#         phrases_in_sent = (
#             occ_df.groupby(["doc_index", "sent_index"])["phrase"]
#             .apply(lambda xs: sorted(set(xs)))  # unique phrases per sentence
#             .rename("phrases")
#             .reset_index()
#         )
#     else:
#         # Fallback to timeline_result.phrase_sentence_df
#         ps = timeline_result.phrase_sentence_df[["doc_index", "sent_index", "phrase"]].copy()
#         phrases_in_sent = (
#             ps.groupby(["doc_index", "sent_index"])["phrase"]
#             .apply(lambda xs: sorted(set(xs)))
#             .rename("phrases")
#             .reset_index()
#         )

#     # Merge with sentence text / timeline index
#     base = timeline_result.sentence_df[["doc_index", "sent_index", "timeline_idx", "sentence_text"]].copy()
#     corpus_flow = phrases_in_sent.merge(base, on=["doc_index", "sent_index"], how="inner")

#     # If you also want dominant_cluster on y-axis, merge doc_topic_profile
#     dtp = doc_topic_profile.reset_index() if isinstance(doc_topic_profile.index, pd.MultiIndex) else doc_topic_profile.copy()
#     corpus_flow = corpus_flow.merge(
#         dtp[["doc_index", "sent_index", "dominant_cluster"]],
#         on=["doc_index", "sent_index"],
#         how="left",
#     )

#     # Keep only sentences that truly have phrases
#     corpus_flow["phrases"] = corpus_flow["phrases"].apply(lambda v: v if isinstance(v, list) else [])
#     corpus_flow = corpus_flow[corpus_flow["phrases"].map(len) > 0].copy()

#     if corpus_flow.empty:
#         return go.Figure().update_layout(title="No sentences with phrases were found.")

#     # -----------------------------
#     # 2) Optional doc filtering
#     # -----------------------------
#     if doc_indices:
#         corpus_flow = corpus_flow[corpus_flow["doc_index"].isin(doc_indices)].copy()
#         if corpus_flow.empty:
#             return go.Figure().update_layout(title="No data for selected documents.")

#     # -----------------------------
#     # 3) Build centrality + count lookup
#     # -----------------------------
#     phrase_info_df = core_result.phrases_df[["phrase", "count"]].copy()

#     if "centrality" in core_result.phrases_df.columns:
#         phrase_info_df["centrality"] = core_result.phrases_df["centrality"].astype(float)
#     elif "x" in core_result.phrases_df.columns and "y" in core_result.phrases_df.columns:
#         cx = float(core_result.phrases_df["x"].mean())
#         cy = float(core_result.phrases_df["y"].mean())
#         phrase_info_df["centrality"] = np.sqrt(
#             (core_result.phrases_df["x"] - cx) ** 2 + (core_result.phrases_df["y"] - cy) ** 2
#         )
#     else:
#         phrase_info_df["centrality"] = 0.0

#     centrality_map = phrase_info_df.set_index("phrase")["centrality"].to_dict()
#     count_map = phrase_info_df.set_index("phrase")["count"].to_dict()

#     # -----------------------------
#     # 4) Pick ONE dominant phrase per sentence (most central among phrases in that sentence)
#     # -----------------------------
#     def get_most_central_phrase(phrases: List[str]) -> str:
#         if not phrases:
#             return "∅"
#         # prefer phrases that exist in map
#         known = [p for p in phrases if p in centrality_map]
#         pool = known if known else phrases
#         return min(pool, key=lambda p: centrality_map.get(p, float("inf")))

#     corpus_flow["top_phrase"] = corpus_flow["phrases"].apply(get_most_central_phrase)

#     # -----------------------------
#     # 5) IMPORTANT: use FULL unique legend key for grouping
#     # -----------------------------
#     corpus_flow["legend_key"] = corpus_flow["top_phrase"].apply(
#         lambda p: f"{p} (C: {count_map.get(p, 0)})"
#     )

#     # Display label is truncated, but NOT used for grouping
#     corpus_flow["legend_display"] = corpus_flow["legend_key"].apply(lambda s: _truncate(s, legend_max_chars))

#     # Sort legend keys by the underlying phrase centrality (ascending = most central first)
#     uniq = (
#         corpus_flow[["top_phrase", "legend_key", "legend_display"]]
#         .drop_duplicates("legend_key")
#         .assign(centrality=lambda d: d["top_phrase"].map(lambda p: centrality_map.get(p, float("inf"))))
#         .sort_values("centrality", ascending=True)
#     )
#     sorted_keys = uniq["legend_key"].tolist()

#     # -----------------------------
#     # 6) Colors: top-N most central phrases get palette, others lightgrey
#     # -----------------------------
#     top_phrases = (
#         pd.Series(centrality_map)
#         .sort_values(ascending=True)
#         .head(top_n_phrases_to_color)
#         .index
#         .tolist()
#     )
#     top_phrases = set(top_phrases)

#     palette = px.colors.qualitative.Plotly
#     color_map: Dict[str, str] = {}
#     i = 0
#     legend_key_to_phrase = uniq.set_index("legend_key")["top_phrase"].to_dict()

#     for k in sorted_keys:
#         phrase = legend_key_to_phrase.get(k, "∅")
#         if phrase in top_phrases:
#             color_map[k] = palette[i % len(palette)]
#             i += 1
#         else:
#             color_map[k] = "lightgrey"

#     # -----------------------------
#     # 7) Y-axis labels (cluster ids or label strings)
#     # -----------------------------
#     has_labels = bool(cluster_name_map)
#     y_axis_label = "dominant_cluster"
#     if has_labels:
#         corpus_flow["topic_label"] = corpus_flow["dominant_cluster"].apply(
#             lambda cid: cluster_name_map.get(int(cid), f"Cluster {cid}") if pd.notna(cid) else "Unknown"
#         )
#         y_axis_label = "topic_label"

#     # -----------------------------
#     # 8) Hover HTML (uses your helpers)
#     # -----------------------------
#     corpus_flow["highlighted_sentence"] = corpus_flow.apply(
#         lambda row: highlight_phrases_in_sentence(row["sentence_text"], row["phrases"]),
#         axis=1,
#     )
#     corpus_flow["hover_html"] = corpus_flow.apply(
#         lambda row: build_hover_html(row, has_labels),
#         axis=1,
#     )

#     # -----------------------------
#     # 9) Plot (color by FULL legend_key)
#     # -----------------------------
#     corpus_flow = corpus_flow.sort_values("timeline_idx")

#     fig = px.scatter(
#         corpus_flow,
#         x="timeline_idx",
#         y=y_axis_label,
#         color="legend_key",  # ✅ FULL KEY (no truncation collisions)
#         color_discrete_map=color_map,
#         category_orders={"legend_key": sorted_keys},
#         title=("Corpus-Wide Thematic Flow" if not doc_indices else f"Thematic Flow for Doc(s): {doc_indices}"),
#         labels={
#             "timeline_idx": "Global Key Sentence Sequence",
#             y_axis_label: "Dominant Topic",
#             "legend_key": "Dominant Phrase",
#         },
#         custom_data=["hover_html"],
#     )

#     fig.update_traces(
#         marker=dict(size=8, opacity=0.8, line=dict(width=1, color="DarkSlateGrey")),
#         hovertemplate="%{customdata[0]}<extra></extra>",
#     )

#     # Rename legend display names AFTER the plot is built (doesn't affect grouping)
#     display_names = _make_unique(uniq["legend_display"].tolist())
#     legend_key_to_display = dict(zip(uniq["legend_key"].tolist(), display_names))
#     fig.for_each_trace(lambda tr: tr.update(name=legend_key_to_display.get(tr.name, tr.name)))

#     # -----------------------------
#     # 10) Document separators
#     # -----------------------------
#     if show_doc_separators and (not doc_indices or len(doc_indices) > 1):
#         bounds = corpus_flow.groupby("doc_index")["timeline_idx"].max().sort_index()
#         for j in range(len(bounds) - 1):
#             next_doc = bounds.index[j + 1]
#             next_min = corpus_flow.loc[corpus_flow["doc_index"] == next_doc, "timeline_idx"].min()
#             boundary_pos = (bounds.iloc[j] + next_min) / 2.0
#             fig.add_vline(
#                 x=boundary_pos,
#                 line_width=1,
#                 line_dash="dash",
#                 line_color="grey",
#                 annotation_text=f"Doc {next_doc} →",
#                 annotation_position="top left",
#             )

#     fig.update_layout(
#         yaxis={"categoryorder": "total ascending"},
#         hoverlabel=dict(bgcolor="white", font_size=13),
#         height=800,
#         margin=dict(l=100),
#     )

#     return fig, corpus_flow




    
    
def create_topic_summary_grid(
    core_result: TopicCoreResult,
    labeling_result: Optional[TopicLabelingResult] = None,
    top_n_phrases: int = 5,
    num_cols: int = 3,
    exclude_noise: bool = False
) -> go.Figure:
    """
    Creates a grid of bar charts summarizing the top phrases for each topic.

    Args:
        core_result: The TopicCoreResult from the modeler
        labeling_result: Optional TopicLabelingResult to add LLM titles
        top_n_phrases: The number of top phrases to show for each topic
        num_cols: The number of columns in the subplot grid
        exclude_noise: Whether to exclude cluster -1 (noise)

    Returns:
        A Plotly Figure object containing the grid
    """
    clusters = sorted(core_result.clusters, key=lambda x: x.cluster_id)
    
    if exclude_noise:
        clusters = [c for c in clusters if c.cluster_id != -1]
    
    if not clusters:
        return go.Figure().update_layout(title="No topics to display.")

    num_rows = math.ceil(len(clusters) / num_cols)
    
    # Get cluster titles from LLM labels if available
    cluster_name_map = labeling_result.cluster_name_map if labeling_result else {}

    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=[cluster_name_map.get(c.cluster_id, f"Cluster {c.cluster_id}") for c in clusters]
    )

    # Prepare a lookup for phrase counts
    phrase_counts = core_result.phrases_df.set_index('phrase')['count']

    for i, cluster in enumerate(clusters):
        row = i // num_cols + 1
        col = i % num_cols + 1

        # Get top N phrases and their counts
        top_phrases = cluster.representative_phrases[:top_n_phrases]
        counts = [phrase_counts.get(p, 0) for p in top_phrases]

        # Use underscores for multi-word phrases and reverse for plotting
        display_phrases = [p.replace(' ', '_') for p in top_phrases][::-1]
        display_counts = counts[::-1]

        fig.add_trace(
            go.Bar(
                y=display_phrases,
                x=display_counts,
                orientation='h',
                marker_color='rgb(55, 83, 109)'
            ),
            row=row, col=col
        )

    fig.update_layout(
        height=250 * num_rows,
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
        title_text="Top Phrases per Topic"
    )
    fig.update_yaxes(tickfont=dict(size=10))
    
    return fig


    
def create_topic_wordcloud_grid(
    core_result: TopicCoreResult,
    labeling_result: Optional[TopicLabelingResult] = None,
    num_cols: int = 2,
    max_phrases_per_cloud: int = 40,
    length_emphasis: float = 0.5,
    exclude_noise: bool = True,
    colormap: str = 'gist_heat',
    mask_array: Optional[np.ndarray] = None
) -> plt.Figure:
    """
    Creates a grid of word clouds summarizing the most semantically important
    phrases for each topic, rendered using Matplotlib.

    The "importance" of a phrase is determined by a tunable formula that balances
    its frequency with its length (number of words). This allows multi-word phrases
    that are highly characteristic of a topic to be visually prominent.

    Args:
        core_result: The main result object from the TopicModeler
        labeling_result: Optional result from TopicLabeler for human-readable titles
        num_cols: The number of columns to arrange the word cloud subplots in
        max_phrases_per_cloud: Maximum number of phrases to include in each cloud
        length_emphasis: Controls importance of phrase length (0.0-1.0+)
            - 0.0: Purely frequency-based
            - 0.5: Balanced (default)
            - 1.0+: Strong preference for longer phrases
        exclude_noise: If True, exclude the noise cluster (-1)
        colormap: Matplotlib colormap name (e.g., 'gist_heat', 'viridis', 'plasma')
        mask_array: Optional NumPy array representing an image mask

    Returns:
        A Matplotlib Figure object containing the grid of word clouds
    """
    # 1. Filter and sort clusters based on user preference
    clusters = sorted(core_result.clusters, key=lambda x: x.cluster_id)
    if exclude_noise:
        clusters = [c for c in clusters if c.cluster_id != -1]

    # Handle the case where there are no topics to display
    if not clusters:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No topics to display.", ha='center', va='center', fontsize=12)
        ax.axis("off")
        return fig

    # 2. Set up the Matplotlib subplot grid
    num_rows = math.ceil(len(clusters) / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 4.5 * num_rows))
    axes = np.array(axes).flatten()  # Flatten for easy iteration

    # Prepare a lookup for LLM-generated titles if available
    cluster_name_map = labeling_result.cluster_name_map if labeling_result else {}

    # 3. Iterate through each cluster to generate and plot a word cloud
    for i, cluster in enumerate(clusters):
        ax = axes[i]
        title = cluster_name_map.get(cluster.cluster_id, f"Cluster {cluster.cluster_id}")

        # Calculate semantic importance for each phrase
        weights = {}
        cluster_phrases = pd.DataFrame(
            zip(cluster.phrases, cluster.phrase_counts),
            columns=['phrase', 'count']
        )
        
        for _, row in cluster_phrases.iterrows():
            phrase, count = row['phrase'], row['count']
            display_phrase = phrase.replace(' ', '_')  # Underscores for readability
            num_words = len(phrase.split())
            
            # Core weighting formula
            weight = count * (num_words ** length_emphasis)
            weights[display_phrase] = weight
        
        # Sort by weight and select the top N phrases for the cloud
        top_weights = dict(sorted(weights.items(), key=lambda item: item[1], reverse=True)[:max_phrases_per_cloud])

        if not top_weights:
            ax.text(0.5, 0.5, "Not enough data", ha='center', va='center')
            ax.set_title(title)
            ax.axis("off")
            continue

        # Generate the word cloud object
        wc = WordCloud(
            background_color="white",
            width=400,
            height=300,
            prefer_horizontal=0.7,
            max_font_size=45,
            min_font_size=15,
            colormap=colormap,
            mask=mask_array,
        ).generate_from_frequencies(top_weights)

        # Plot the generated image
        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(title)
        ax.axis("off")

    # 4. Clean up any unused subplots in the grid
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.tight_layout(pad=2.0)
    return fig
