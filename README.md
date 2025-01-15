## Description
The task is to implement a prototype tool that automates the comparison
of control descriptions (provided as text) with regulatory requirements outlined in a .pdf document. The goal is to identify potential gaps or mismatches between the controls and the regulatory requirements by leveraging local Large Language Models for semantic understanding and text comparison.

## Setup

Download the contents of the repository, either manually or via git clone:
```bash
git clone https://github.com/skrap17/rhizon
```

Open terminal and navigate to the folder where the files have been downloaded. Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the necessary packages:

```bash
pip install ./requirements.txt
```

If you want to run the Jupyter notebook, you might need to install additional packages:

```bash
pip install jupyter
```
And depending on your IDE you might also need to install an extension that allows to run Jupyter notebooks.

## Project structure
```bash
.
├── documents               # Input files
├── models                  # Translation and embedding models for local use
├── language_models.py      # Utilities for translation and embedding   
├── parser.py               # Utilities for loading the data 
├── README.md
├── requirements.txt
├── solution_app.py         # Streamlit app with the solution
└── solution.ipynb          # Jupyter notebook with a step by step solution    
```

## Approach
The basic idea is to extract the control descriptions and regulatory articles and then embed them into a latent vector space using a transformer-based encoder. After that compare every control-article pair of embeddings using to produce a list of similarity scores. The elements of the pair are decided to be relevant to one another if their similarity is greater than a certain threshold.

### Translation
The first challenge is that the regulatory document is in German, while the controls are in English. Given the fact that the latter are much shorter, it would be efficient to translate them to German. However, then we would require a good embedding model trained on German. Therefore, given the variety of models available for English, it was decided to instead translate the regulatory articles to English using a transformer-based translation model. It is more resource intensive and there might be artifacts created by the translation process, but in my opinion these downsides are neglected by the greater choice of English models that can be applied later (including generative models if one decides to expand upon the current approach).

### Similarity thresholds
Firstly, the vector similarity metric used is cosine similarity. To find a potentially suitable threshold, we can look at the distribution of scores of all pairs. After experimenting with some values it was decided to use a 0.8 quantile of this distribution as a threshold.

Additionally, once we establish which pairs are similar, we need to determine which articles aren't sufficiently covered by the available controls. To do this we can calculate the total number of relevant controls of each article and compare it to the average number of relevant controls across all articles. The article is deemed to be covered insufficiently if it covered less than on average. It is a heuristic criterion and other approaches can be used as well depending on the exact requirements.

### Input parsing
A separate but necessary problem is to correctly parse and split the inputs. With controls it is quite straightforward since they were specifically manufactured in a structured manner for this assignment. With regulatory articles it is a bit different and some liberties of interpretation were taken.

First, we ignore the cover, table of contents and the appendix, leaving only the main body. We assume that the test is structured in 3 levels of hierarchy with chapters, sections and subsections. We also assume that the formatting is consistent across the document (font sizes, styles), so that the structural elements can be detected with a simple pattern. For instance:
1. Chapters are assumed to have a bold font, that is larger than usual.
2. Sections are assumed to have a regular font, that is larger than usual.
3. Subsections are assumed to have a bold font, that is of regular size.

Moreover, each stretch of text defined by the triplet (chapter, section, subsection) is assumed to be self-contained and thus serves as full article whose coverage will be determined independently.

### Models used
After some experimentation the following models have been chosen to use for this project:
1. `Helsinki-NLP/opus-mt-de-en` - for translation due to its small weight and pretty robust results.
2. `Alibaba-NLP/gte-base-en-v1.5` - for embedding because of its lightness, decent MTEB leaderboard ranking (for such a compact model) and big maximum context window of 8192 tokens.