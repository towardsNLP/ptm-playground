# 🧭 PhraseTopicMiner Playground

An interactive Streamlit application for exploring [PhraseTopicMiner](https://github.com/towardsNLP/PhraseTopicMiner), a phrase-centric topic modeling library.

**Try the Live Demo:** [Coming Soon]

---

## ✨ Features

- **📁 Flexible Data Input:** Use example corpus, paste text, or upload your own files
- **⚙️ Full Parameter Control:** Configure every aspect of the PTM pipeline
- **🗺️ Interactive Visualizations:** 
  - Semantic topic maps with DataMapPlot
  - Thematic flow charts across documents
  - Word clouds and bar charts for topic summaries
- **🔍 Deep Analysis:**
  - Sub-topic discovery
  - Phrase importance scoring
  - Document-topic profiling
- **📊 Export Everything:**
  - Download visualizations as HTML/PNG
  - Export data as CSV/DataFrames
  - Generate reproducible Python code
- **🏷️ Optional LLM Labeling:** Generate human-readable topic names with GPT

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/towardsNLP/ptm-playground.git
cd ptm-playground

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run the app
streamlit run app_refactored.py
```

### Using the Playground

1. **Load Data:** Choose from example corpus, paste text, or upload files
2. **Configure:** Adjust phrase mining and topic modeling parameters (or use defaults)
3. **Run Analysis:** Click "Mine Topics" and wait for results
4. **Explore:** Scroll through interactive visualizations and insights
5. **Export:** Download visualizations, data, and code snippets

---

## 📂 Project Structure

```
ptm-playground/
├── app_refactored.py           # Main Streamlit application
│
├── src/                         # Source modules
│   ├── __init__.py              # Package initialization
│   ├── document_loader.py       # Document loading & metadata extraction
│   ├── analysis.py              # Topic analysis functions
│   ├── visualizations.py        # Plotting functions
│   ├── pipeline.py              # Cached PTM pipeline
│   └── utils.py                 # Helper utilities
│
├── examples/
│   └── sample_corpus/           # Example philosophical dialogues
│       ├── 01_epistemology_rationalism_empiricism.txt
│       ├── 02_ethics_consequentialism_deontology.txt
│       ├── 03_metaphysics_materialism_idealism.txt
│       ├── 04_political_philosophy_liberty_justice.txt
│       ├── 05_philosophy_mind_consciousness.txt
│       └── README.md
│
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml
```

---

## 🎯 Key Concepts

### Phrase-Centric Topic Modeling

Unlike traditional topic models (LDA, BERTopic) that operate on single words, PhraseTopicMiner:

1. **Extracts meaningful multi-word phrases** using POS patterns
2. **Clusters phrases semantically** using embeddings + HDBSCAN/KMeans
3. **Provides full provenance** with document IDs, page numbers, and citations
4. **Enables temporal analysis** via timeline construction

### Example Workflow

```python
import phrasetopicminer as ptm

# 1. Initialize miner
miner = ptm.PhraseMiner(spacy_model="en_core_web_sm")
_, _, phrase_records, sentences_by_doc = miner.mine_phrases_with_types(docs)

# 2. Build topic model
modeler = ptm.TopicModeler(embedding_model="all-MiniLM-L6-v2")
core_result = modeler.fit_core(
    phrase_records=phrase_records,
    sentences_by_doc=sentences_by_doc,
    include_patterns={"BaseNP", "NP+PP"},
    min_freq_bigram=2
)

# 3. Build timeline
timeline_builder = ptm.TopicTimelineBuilder()
timeline_result = timeline_builder.build(core_result, sentences_by_doc)
```

---

## ⚙️ Configuration Options

### Phrase Mining

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | `"spacy"` | POS-tagging method (`spacy` or `nltk`) |
| `spacy_model` | str | `"en_core_web_sm"` | spaCy model for POS tagging |
| `include_verb_phrases` | bool | `False` | Extract verb phrases in addition to NPs |
| `clean_markdown` | bool | `True` | Remove markdown syntax from text |

### Topic Modeling

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_backend` | str | `"sentence_transformers"` | Embedding method |
| `embedding_model` | str | `"all-MiniLM-L6-v2"` | Model for sentence embeddings |
| `include_patterns` | set | `{"BaseNP", "NP+PP", ...}` | Phrase patterns to include |
| `min_freq_unigram` | int | `3` | Min frequency for 1-word phrases |
| `min_freq_bigram` | int | `2` | Min frequency for 2-word phrases |
| `min_freq_trigram_plus` | int | `1` | Min frequency for 3+ word phrases |
| `pca_n_components` | int | `50` | Number of PCA components (0 to skip) |
| `clustering_algorithm` | str | `"hdbscan"` | Clustering method (`hdbscan` or `kmeans`) |
| `hdbscan_min_cluster_size` | int | `5` | Min phrases per cluster |

### Timeline

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeline_mode` | str | `"reading_time"` | Timeline construction mode |
| `speech_rate_wpm` | int | `160` | Reading speed (words per minute) |

---

## 📊 Output Formats

### DataFrames

All analysis results are available as pandas DataFrames:

- **`core_result.phrases_df`**: All phrases with clusters, counts, embeddings
- **`timeline_result.sentence_df`**: Sentence-level data with timeline indices
- **`phrase_importance_df`**: Phrase importance scores (frequency × length × centrality)

### Visualizations

All visualizations support download:

- **Interactive Topic Map (HTML)**: Explorable 2D semantic space
- **Thematic Flow Chart (HTML)**: Plotly interactive scatter plot
- **Topic Summaries (HTML/PNG)**: Bar charts or word clouds
- **Python Code (TXT)**: Reproducible analysis script

---

## 🎓 Example Corpus

The playground includes 5 synthetic philosophical dialogues (6,624 words) covering:

1. **Epistemology** - Rationalism vs Empiricism, Kant's synthesis
2. **Ethics** - Consequentialism, Deontology, Virtue Ethics
3. **Metaphysics** - Materialism, Idealism, Consciousness
4. **Political Philosophy** - Liberty, Justice, Social Contract
5. **Philosophy of Mind** - Hard Problem, Qualia, Functionalism

These dialogues demonstrate PTM's ability to:
- Extract technical philosophical concepts
- Cluster related ideas semantically
- Provide evidence trails with citations

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## 📝 Citation

If you use PhraseTopicMiner in your research, please cite:

```bibtex
@software{phrasetopicminer2024,
  title={PhraseTopicMiner: Phrase-Centric Topic Modeling with Full Provenance},
  author={Ahmad, Your Name},
  year={2024},
  url={https://github.com/towardsNLP/PhraseTopicMiner}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

---

## 🔗 Links

- **PhraseTopicMiner Library:** https://github.com/towardsNLP/PhraseTopicMiner
- **Documentation:** [Coming Soon]
- **Live Demo:** [Coming Soon]
- **Issues:** https://github.com/towardsNLP/ptm-playground/issues

---

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Embeddings from [SentenceTransformers](https://www.sbert.net/)
- POS tagging with [spaCy](https://spacy.io/)
- Clustering with [HDBSCAN](https://hdbscan.readthedocs.io/)
- Visualizations with [Plotly](https://plotly.com/) and [Matplotlib](https://matplotlib.org/)

---

**Made with ❤️ by TowardsNLP**



