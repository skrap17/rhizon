import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from parser import read_controls, read_articles
from language_models import translate, embed
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel

st.title('Rhizon.ai case study')
approach = st.markdown(
'''
#### Approach
The basic idea is to extract the control descriptions and regulatory articles and then embed them into a latent vector space using a transformer-based encoder. After that compare every control-article pair of embeddings using to produce a list of similarity scores. The elements of the pair are decided to be relevant to one another if their similarity is greater than a certain threshold.

##### Translation
The first challenge is that the regulatory document is in German, while the controls are in English. Given the fact that the latter are much shorter, it would be efficient to translate them to German. However, then we would require a good embedding model trained on German. Therefore, given the variety of models available for English, it was decided to instead translate the regulatory articles to English using a transformer-based translation model. It is more resource intensive and there might be artifacts created by the translation process, but in my opinion these downsides are neglected by the greater choice of English models that can be applied later (including generative models if one decides to expand upon the current approach).

##### Similarity thresholds
Firstly, the vector similarity metric used is cosine similarity. To find a potentially suitable threshold, we can look at the distribution of scores of all pairs. After experimenting with some values it was decided to use a 0.8 quantile of this distribution as a threshold.

Additionally, once we establish which pairs are similar, we need to determine which articles aren't sufficiently covered by the available controls. To do this we can calculate the total number of relevant controls of each article and compare it to the average number of relevant controls across all articles. The article is deemed to be covered insufficiently if it covered less than on average. It is a heuristic criterion and other approaches can be used as well depending on the exact requirements.

##### Input parsing
A separate but necessary problem is to correctly parse and split the inputs. With controls it is quite straightforward since they were specifically manufactured in a structured manner for this assignment. With regulatory articles it is a bit different and some liberties of interpretation were taken.

First, we ignore the cover, table of contents and the appendix, leaving only the main body. We assume that the test is structured in 3 levels of hierarchy with chapters, sections and subsections. We also assume that the formatting is consistent across the document (font sizes, styles), so that the structural elements can be detected with a simple pattern. For instance:
1. Chapters are assumed to have a bold font, that is larger than usual.
2. Sections are assumed to have a regular font, that is larger than usual.
3. Subsections are assumed to have a bold font, that is of regular size.

Moreover, each stretch of text defined by the triplet (chapter, section, subsection) is assumed to be self-contained and thus serves as full article whose coverage will be determined independently.

##### Models used
After some experimentation the following models have been chosen to use for this project:
1. `Helsinki-NLP/opus-mt-de-en` - for translation due to its small weight and pretty robust results.
2. `Alibaba-NLP/gte-base-en-v1.5` - for embedding because of its lightness, decent MTEB leaderboard ranking (for such a compact model) and big maximum context window of 8192 tokens.
'''
)

document_path = './documents'
control_path = f'{document_path}/Internal_Controls.docx'
articles_path = f'{document_path}/finma rs 2023 01 20221207.pdf'
models_path = './models'

translation_model_name = 'Helsinki-NLP/opus-mt-de-en'
translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_name)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(f'{models_path}/{translation_model_name}')

embedding_model_name = 'Alibaba-NLP/gte-base-en-v1.5'
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(f'{models_path}/{embedding_model_name}', trust_remote_code=True, add_pooling_layer=False)
embedding_model.eval()

data_load_state = st.text('Loading data...')
df_controls = read_controls(control_path)
df_articles = read_articles(articles_path) 
data_load_state.text('Loading data...done!')

st.subheader('Controls')
st.write(df_controls)

translation_state = st.text('Translating articles...(this can take a few of minutes, especially whe running in Streamlit community cloud)')
df_articles = translate(df_articles, 'text', translation_tokenizer, translation_model)
translation_state.text('Translating articles...done!')

st.subheader('Articles')
st.write(df_articles)

embedding_state = st.text('Embedding data...')
df_controls = embed(df_controls, 'text', embedding_tokenizer, embedding_model, is_query=False)
df_articles = embed(df_articles, 'translated_text', embedding_tokenizer, embedding_model, is_query=False)
embedding_state.text('Embedding data...done!')

similarities = np.array(df_articles['translated_text_embedding'].to_list()) @ np.array(df_controls['text_embedding'].to_list()).T
average = similarities.mean()
percentage = 0.8
threshold = np.quantile(similarities, percentage)

st.subheader('Distribution of similarities between controls and articles')
fig1, ax1 = plt.subplots()
ax1.hist(similarities.flatten(), bins=20)
ax1.axvline(average, c='r', linestyle='--', label='Average')
ax1.axvline(threshold, c='orange', linestyle='--', label=f'{percentage}-quantile')
ax1.legend()
st.pyplot(fig1)

st.subheader('\nSimilarity score between control descriptions and artices')
fig2, ax2 = plt.subplots()
im = ax2.imshow(similarities)
ax2.set_xticks(range(len(df_controls)), labels=df_controls['id'], rotation=45, ha='right', rotation_mode='anchor')
ax2.set_yticks(range(len(df_articles)), labels=df_articles['id'])
for i in range(len(df_articles)):
    for j in range(len(df_controls)):
        text = ax2.text(j, i, np.round(similarities[i, j], 2), ha='center', va='center', color='w', size=7.5)
st.pyplot(fig2)

binary_similarities = np.where(similarities > threshold, 1, 0)

st.subheader('\nBinary similarity score between control descriptions and artices (Similar or not similar)')
st.text(f'It is determined by setting all similarities below a threshold to 0 and above to 1. The threshold is selected to be {percentage}-quantile of the distribution of all similarities.')
fig3, ax3 = plt.subplots()
im = ax3.imshow(binary_similarities)
ax3.set_xticks(range(len(df_controls)), labels=df_controls['id'], rotation=45, ha='right', rotation_mode='anchor')
ax3.set_yticks(range(len(df_articles)), labels=df_articles['id'])
st.pyplot(fig3)

average_coverage = binary_similarities.sum(axis=1).mean()
df_articles['number_of_controls'] = binary_similarities.sum(axis=1) 
df_articles['is_covered'] =  (df_articles['number_of_controls'] >= average_coverage)
poorly_covered_articles = df_articles[~df_articles['is_covered']]
st.subheader('\nArticles that are poorly covered by controls')
poorly_covered_desc = st.text(
'An article is determined to be insufficiently covered if its total number of relevant controls is less than the average across all articles.'
)
st.write(poorly_covered_articles[['chapter', 'section', 'subsection', 'number_of_controls']])

st.subheader('\nMapping between controls and most relevant articles')
df_controls['relevant_article_indexes'] = [[i for i, bin_sim in enumerate(bin_sim_list) if bin_sim] for bin_sim_list in binary_similarities.T]
df_controls['relevant_articles'] = df_controls['relevant_article_indexes'].apply(
    lambda x: df_articles.iloc[x]['id'].to_list()
)
df_controls['number_of_relevant_articles'] = df_controls['relevant_articles'].map(len)
st.write(df_controls[['id', 'relevant_articles', 'number_of_relevant_articles']])

st.subheader('Conclusions')
conclusions = st.markdown(
'''
Though the results look reasonable, a more thorough analysis is needed to determine whether the final relevancy relationships indeed make sense. In order to do so, ideally a closer examination of the content of each article by a expert in the regulatory domain is required, to find false positives and false negatives. Upon this inspection the following courses of action may be taken:
1. Tune the similarity threshold that determines relevancy
2. Explore a different embedding model (perhaps one specifically trained on this domain)
3. Combine multiple vector modes to get a fusion score (eg add a lexical model like TF-IDF)
4. Implement a chunking strategy to ensure that facts don't get dilluted by long context in some articles
5. Rethink the approach entirely:
    - Translate the control descriptions to German and find a German embedding model
    - For every article rephrase it using a generative model to achieve higher vocabulary and grammar variability and run the similarity search on every version of the article text
    - Rely completely on a generative model by passing each article - control pair in a prompt asking to determine whether they are relevant to one another.
'''
)