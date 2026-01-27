"""
Document loader with comprehensive metadata extraction.

This module provides functions to load documents from a directory and extract
detailed metadata for use in both the PTM Playground and TopicTrace Studio.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import re


def load_sample_documents(path: str) -> Dict[str, Any]:
    """
    Load text documents from a directory and extract comprehensive metadata.
    
    Args:
        path: Path to directory containing .txt files
        
    Returns:
        Dictionary containing:
        - documents: List[str] - Raw text content of each document
        - metadata: List[dict] - Per-document metadata (doc_id, filename, stats)
        - corpus_stats: dict - Corpus-level statistics
        
    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If no .txt files found in directory
    """
    path_obj = Path(path)
    
    # Validate path
    if not path_obj.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    
    if not path_obj.is_dir():
        raise ValueError(f"Path is not a directory: {path}")
    
    # Get all .txt files, sorted alphabetically
    txt_files = sorted(path_obj.glob("*.txt"))
    
    if not txt_files:
        raise ValueError(f"No .txt files found in directory: {path}")
    
    # Load documents and extract metadata
    documents = []
    metadata = []
    
    for doc_id, filepath in enumerate(txt_files):
        try:
            # Read document content
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            documents.append(content)
            
            # Extract document metadata
            doc_metadata = {
                'doc_id': doc_id,
                'filename': filepath.name,
                'filepath': str(filepath),
                'word_count': _count_words(content),
                'char_count': len(content),
                'char_count_no_spaces': len(content.replace(' ', '')),
                'sentence_count': _count_sentences(content),
                'paragraph_count': _count_paragraphs(content),
                'avg_words_per_sentence': _avg_words_per_sentence(content),
            }
            
            metadata.append(doc_metadata)
            
        except Exception as e:
            print(f"Warning: Failed to load {filepath.name}: {e}")
            continue
    
    # Calculate corpus-level statistics
    corpus_stats = _calculate_corpus_stats(metadata)
    
    return {
        'documents': documents,
        'metadata': metadata,
        'corpus_stats': corpus_stats
    }


def _count_words(text: str) -> int:
    """Count words in text (splits on whitespace)."""
    return len(text.split())


def _count_sentences(text: str) -> int:
    """
    Approximate sentence count using simple heuristic.
    Counts periods, exclamation marks, and question marks.
    """
    # Simple heuristic: split on sentence-ending punctuation
    sentences = re.split(r'[.!?]+', text)
    # Filter out empty strings
    sentences = [s for s in sentences if s.strip()]
    return len(sentences)


def _count_paragraphs(text: str) -> int:
    """Count paragraphs (separated by blank lines)."""
    paragraphs = text.split('\n\n')
    # Filter out empty paragraphs
    paragraphs = [p for p in paragraphs if p.strip()]
    return len(paragraphs)


def _avg_words_per_sentence(text: str) -> float:
    """Calculate average words per sentence."""
    word_count = _count_words(text)
    sentence_count = _count_sentences(text)
    
    if sentence_count == 0:
        return 0.0
    
    return round(word_count / sentence_count, 1)


def _calculate_corpus_stats(metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate corpus-level statistics from document metadata."""
    if not metadata:
        return {}
    
    total_docs = len(metadata)
    total_words = sum(doc['word_count'] for doc in metadata)
    total_chars = sum(doc['char_count'] for doc in metadata)
    total_sentences = sum(doc['sentence_count'] for doc in metadata)
    total_paragraphs = sum(doc['paragraph_count'] for doc in metadata)
    
    # Per-document averages
    avg_words_per_doc = round(total_words / total_docs, 1)
    avg_sentences_per_doc = round(total_sentences / total_docs, 1)
    avg_paragraphs_per_doc = round(total_paragraphs / total_docs, 1)
    
    # Find min/max document by word count
    min_doc = min(metadata, key=lambda x: x['word_count'])
    max_doc = max(metadata, key=lambda x: x['word_count'])
    
    return {
        'total_documents': total_docs,
        'total_words': total_words,
        'total_characters': total_chars,
        'total_sentences': total_sentences,
        'total_paragraphs': total_paragraphs,
        'avg_words_per_doc': avg_words_per_doc,
        'avg_sentences_per_doc': avg_sentences_per_doc,
        'avg_paragraphs_per_doc': avg_paragraphs_per_doc,
        'min_doc_words': min_doc['word_count'],
        'min_doc_filename': min_doc['filename'],
        'max_doc_words': max_doc['word_count'],
        'max_doc_filename': max_doc['filename'],
    }


def format_corpus_stats_for_display(corpus_stats: Dict[str, Any]) -> str:
    """
    Format corpus statistics for display in Streamlit or console.
    
    Args:
        corpus_stats: Dictionary returned from load_sample_documents()['corpus_stats']
        
    Returns:
        Formatted string with corpus statistics
    """
    if not corpus_stats:
        return "No corpus statistics available."
    
    lines = [
        f"📚 **Corpus Statistics**",
        f"",
        f"Total Documents: {corpus_stats['total_documents']}",
        f"Total Words: {corpus_stats['total_words']:,}",
        f"Total Characters: {corpus_stats['total_characters']:,}",
        f"Total Sentences: {corpus_stats['total_sentences']:,}",
        f"",
        f"**Averages per Document:**",
        f"- Words: {corpus_stats['avg_words_per_doc']}",
        f"- Sentences: {corpus_stats['avg_sentences_per_doc']}",
        f"- Paragraphs: {corpus_stats['avg_paragraphs_per_doc']}",
        f"",
        f"**Document Range:**",
        f"- Smallest: {corpus_stats['min_doc_filename']} ({corpus_stats['min_doc_words']} words)",
        f"- Largest: {corpus_stats['max_doc_filename']} ({corpus_stats['max_doc_words']} words)",
    ]
    
    return "\n".join(lines)


# Example usage (for testing)
if __name__ == "__main__":
    # Test with sample corpus
    try:
        result = load_sample_documents("/home/claude/examples/sample_corpus")
        
        print("=" * 60)
        print("DOCUMENT LOADER TEST")
        print("=" * 60)
        print()
        
        print(format_corpus_stats_for_display(result['corpus_stats']))
        print()
        
        print("=" * 60)
        print("DOCUMENT METADATA")
        print("=" * 60)
        for doc_meta in result['metadata']:
            print(f"\nDoc {doc_meta['doc_id']}: {doc_meta['filename']}")
            print(f"  Words: {doc_meta['word_count']}")
            print(f"  Sentences: {doc_meta['sentence_count']}")
            print(f"  Avg words/sentence: {doc_meta['avg_words_per_sentence']}")
        
    except Exception as e:
        print(f"Error: {e}")
