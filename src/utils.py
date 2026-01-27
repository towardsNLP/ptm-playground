"""
Utility Functions for PTM Playground

This module contains helper functions for text processing, HTML generation,
and other common operations used throughout the playground.
"""

import re
import html
import textwrap
import pandas as pd
from typing import List


def wrap_html(text: str, width: int = 80) -> str:
    """
    Wrap long text for hover displays by inserting <br> between words.
    
    Args:
        text: Text to wrap
        width: Maximum width in characters
        
    Returns:
        HTML string with <br> tags for wrapping
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    lines = textwrap.wrap(
        text,
        width=width,
        break_long_words=False,
        replace_whitespace=False,
    )
    return "<br>".join(lines)


def highlight_phrases_in_sentence(sentence: str, phrases: List[str]) -> str:
    """
    Highlights a list of phrases within a sentence string for HTML display.
    Handles overlapping phrases by highlighting the longest ones first.

    Args:
        sentence: The original sentence text
        phrases: A list of phrases to highlight

    Returns:
        HTML string with phrases wrapped in <b><i> tags
    """
    if not isinstance(sentence, str) or not sentence.strip():
        return ""

    # Escape once to avoid injection, then work on the escaped text
    text = html.escape(sentence)

    # Clean & sort phrases longest-first to avoid nested matching issues
    cleaned = {
        p for p in phrases
        if isinstance(p, str) and p.strip()
    }
    sorted_phrases = sorted(cleaned, key=len, reverse=True)

    for phrase in sorted_phrases:
        esc_phrase = html.escape(phrase)
        # whole-word, case-insensitive
        pattern = re.compile(rf'\b({esc_phrase})\b', flags=re.IGNORECASE)
        text = pattern.sub(r'<b><i>\1</i></b>', text)

    return text


def build_hover_html(row: pd.Series, has_labels: bool) -> str:
    """
    Compose the full HTML for the hover label of a single point in a plot.
    
    Args:
        row: DataFrame row with point data
        has_labels: Whether LLM-generated labels are available
        
    Returns:
        HTML string for hover display
    """
    # if has_labels:
    #     header = row["topic_label"]
    # else:
    #     header = f"Cluster {row['dominant_cluster']}"

    # # Wrap the highlighted sentence so the popup doesn't get super wide
    # wrapped_sentence = wrap_html(row["highlighted_sentence"], width=80)

    # return (
    #     f"<b>{html.escape(str(header))}</b><br>"
    #     f"Dominant phrase: <b>{html.escape(str(row['top_phrase']))}</b><br>"
    #     f"Doc: {row['doc_index']} | Global idx: {row['timeline_idx']}<br><br>"
    #     f"{wrapped_sentence}"
    # )
    html_parts = [
        f"<b>Doc {row['doc_index']}, Sent {row['sent_index']}</b><br>",
    ]
    
    if has_labels:
        full_label = row.get('topic_label_full', row.get('topic_label', ''))
        html_parts.append(f"<b>Topic:</b> {full_label}<br>")
    else:
        html_parts.append(f"<b>Cluster:</b> {row['dominant_cluster']}<br>")
    
    # Use FULL phrase label in hover
    full_phrase = row.get('legend_label_full', row.get('legend_label', ''))
    html_parts.append(f"<b>Dominant Phrase:</b> {full_phrase}<br>")
    html_parts.append(f"<b>Sentence:</b> {row['highlighted_sentence']}")
    
    return "".join(html_parts)


def get_phrase_count(phrase: str, phrase_info_df: pd.DataFrame) -> int:
    """
    Get the count for a specific phrase from phrase info dataframe.
    
    Args:
        phrase: The phrase to look up
        phrase_info_df: DataFrame with phrase counts
        
    Returns:
        Count of the phrase
    """
    result = phrase_info_df.loc[phrase_info_df["phrase"] == phrase, "count"]
    return result.item() if not result.empty else 0
