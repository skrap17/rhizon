from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
import pandas as pd
import torch
import streamlit as st

def split_text_into_chunks(text: str, max_tokens: int, tokenizer: AutoTokenizer) -> list[str]:
    """
    Given a string of text, limit on tokens and a tokenizer splits the string into chunks 
    such that each chunk has at most max_tokens. The text can be split only at the end of sentences
    Arguments:
        text (str): string of text to be split into chunks
        max_tokens (int): maximum nember of tokens in each chunk
        tokenizer (AutoTokenizer): an object that converts strings to tokens
    Returns:
        chunks (list(str)): a list of resulting chunks
    """
    sentences = text.split(".")  # Split by sentences
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip() + "."
        # Check if adding this sentence exceeds the token limit
        if len(tokenizer.encode(current_chunk + sentence)) > max_tokens:
            chunks.append(current_chunk.strip())  # Save the current chunk
            current_chunk = sentence  # Start a new chunk
        else:
            current_chunk += " " + sentence

    if current_chunk:  # Add the last chunk
        chunks.append(current_chunk.strip())

    return chunks

@st.cache_data
def translate(df: pd.DataFrame, key: str, _translation_tokenizer: AutoTokenizer, _translation_model: AutoModelForSeq2SeqLM,) -> pd.DataFrame:
    """
    Given a dataframe df translates the text in one of its columns using a given translation model.
    It is up to user to make sure that the model is suitable for the text to be translated.
    Arguments:
        df (pd.DataFrame): dataframe containing the text that needs to be translated
        key (str): name of the column in df containing the text that needs to be translated (make sure it exists!)
        translation_tokenizer (AutoTokenizer): an object that converts strings to tokens
        translation_model (AutoModelForSeq2SeqLM): a translation model
    Returns:
        df (pd.DataFrame): a copy of the original df with an additional column containing translated text
    """
    df[f'{key}_chunks'] = df[key].apply(split_text_into_chunks, args=(_translation_tokenizer.model_max_length // 2, _translation_tokenizer))
    # df[f'{key}_tokens'] = df[f'{key}_chunks'].apply(lambda x: [len(translation_tokenizer.encode(xi)) for xi in x])
    df[f'translated_{key}'] = df[f'{key}_chunks'].apply(
        lambda chunks: ''.join([
            _translation_tokenizer.decode(
                _translation_model.generate(
                    _translation_tokenizer.encode(
                        chunk,
                        return_tensors="pt"
                    ),
                    num_beams=4,
                    early_stopping=True
                )[0],
                skip_special_tokens=True
            )
            for chunk in chunks
        ])
    )
    df = df.drop(columns=[f'{key}_chunks'])
    return df

@st.cache_data
def embed(df: pd.DataFrame, key: str, _embedding_tokenizer: AutoTokenizer, _embedding_model: AutoModel, is_query: bool=False) -> pd.DataFrame:
    """
    A function to embed the text stored in a dataframe into a latent vector space
    Arguments:
        df (pd.DataFrame): dataframe containing the text that needs to be embedded
        key (str): name of the column in df containing the text that needs to be embedded (make sure it exists!)
        embedding_tokenizer (AutoTokenizer): an object that converts strings to tokens
        embedding_model (AutoModelForSeq2SeqLM): an embedding model
        is_query (bool): a flag to indicate whether embedded texts are intended to be queries
    Returns:
        df (pd.DataFrame): a copy of the original df with an additional column containing vector embeddings
    """
    query_prefix = 'Represent this query for searching relevant passages: '
    if is_query:
        texts = df[key].apply(lambda s: query_prefix + s)
    else:
        texts = df[key]
    texts = texts.to_list()

    tokens = _embedding_tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=2048)
    with torch.no_grad():
        embeddings = _embedding_model(**tokens)[0][:, 0]
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    # return embeddings
    df[f'{key}_embedding'] = list(embeddings.numpy())
    return df

