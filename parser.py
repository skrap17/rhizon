from docx import Document
import pdfplumber
import pandas as pd
import streamlit as st


font_size_config = {
    'chapter_size': 12,
    'normal_size': 10
}

relevant_pages_config = {
    'start': 3,
    'end': 15
}

@st.cache_data
def read_controls(fname: str) -> pd.DataFrame:
    """
    Reads the control descriptions from a .docx file.
    Assumes the following structure:
        | Some text
        |
        | 1. Control-ID: id_1
        | Control Name: name_1
        | Description: desc_1
        | ..................
        | n. Control-ID: id_n
        | Control Name: name_n
        | Description: desc_n

    Arguments:
        fname (str): path to the file with control descriptions

    Returns:
        pd.DataFrame: pandas dataframe with columns "id" and "text".
        First column contains control ids, second - concatenated names and descriptions
    """
    doc = Document(fname)
    controls = []

    for paragraph in doc.paragraphs:
        # In order to skip all introductory text we look for 'Control-ID' keyword
        if 'Control-ID' in paragraph.text:
            text = paragraph.text.split('\n') # Get the lines with id, name and description separately
            control_id = text[0][len('Control-ID:'):].strip() # Extract control id
            controls.append({
                'id': control_id, 
                'text': text[1][len('Control Name:'):].strip() + '. ' + text[2][len('Description: '):].strip()
                }) # extract name and description, put them together and save that with id in a dictionary
    return pd.DataFrame.from_records(controls) # convert to a dataframe

@st.cache_data
def read_articles(fname: str) -> pd.DataFrame:
    """
    Read articles from a .pdf

    Assumes that document is structured in chapters, sections and subsections.
        - Chapter format: "Roman numeral. Chapter name". Bold text of bigger size
        - Section format: "Capiital latin letter. Section name". Non-bold text of bigger size
        - Sub-section format: "Lowercase latin letter). Sub-section name". Bold text of regular size
    Font sizes are defined in font_size_config.

    Skips some pages at the start and at the end of the file. Relevant pages are defined in relevant_pages_config.

    Arguments:
        fname (str): path to the file with articles
    
    Returns: 
        pd.DataFrame: pandas dataframe with columns "id", "chapter", "section", "subsection" and "text".
    """
    pdf = pdfplumber.open(fname) # open the pdf
    records = []
    # iterate over all pages
    for page_number, page in enumerate(pdf.pages):
        extracted_lines = page.extract_text_lines(return_chars=True, strip=True) # parse each page
        for line_number in range(len(extracted_lines)):
            extracted_lines[line_number]['page'] = page_number
            records.append(extracted_lines[line_number]) # record parsed text line by line
    
    df = pd.DataFrame.from_records(records) # convert extracted lines into a dataframe
    df = df.drop(columns=['x0', 'x1', 'top', 'bottom']) # drop extra information
    # leave only the relevant pages (skip toc and appendixes)
    df = df[(df['page'] >= relevant_pages_config['start'] - 1) & (df['page'] <= relevant_pages_config['end'] - 1)]

    # function to extract all unique fontsizes present in a line
    def get_fontsizes(row):
        fontsizes = list(set([int(round(char['size'])) for char in row['chars']]))
        return fontsizes

    # function to count the number of bolded characters in a line
    def bold_count(row):
        count = 0
        for char in row['chars']:
            if 'bold' in char['fontname'].lower():
                count += 1
        return count  

    # function to count the number of characters in a line that a smaller than the normal size
    def small_indexes(row):
        return [i for i, char in enumerate(row['chars']) if int(round(char['size'])) < font_size_config['normal_size']]

    # function to delete characters that a smaller than the normal size from a line
    def replace_small(row):
        compact_to_actual = {}
        ci = 0
        for i, char in enumerate(row['text']):
            if char != ' ':
                compact_to_actual[ci] = i
                ci += 1
        actual_indexes = {compact_to_actual[ci] for ci in row['removable']}
        return "".join(char for i, char in enumerate(row['text']) if i not in actual_indexes)

    # function to check if the line follows the chapter format
    def is_chapter(row):
        return row['total'] == row['bolds'] and len(row['fontsize']) == 1 and row['fontsize'][0] == font_size_config['chapter_size']

    # function to check if the line follows the section format
    def is_section(row):
        return row['bolds'] == 0 and len(row['fontsize']) == 1 and row['fontsize'][0] == font_size_config['chapter_size']

    # function to check if the line follows the subsection format
    def is_subsection(row):
        return row['total'] == row['bolds'] and len(row['fontsize']) == 1 and row['fontsize'][0] == font_size_config['normal_size']

    # create additional columns based on helper functions from above
    df['fontsize'] = df.apply(get_fontsizes, axis=1)
    df['total'] = df['chars'].map(len)
    df['bolds'] = df.apply(bold_count, axis=1)
    df['removable'] = df.apply(small_indexes, axis=1) 
    df['text'] = df.apply(replace_small, axis=1) 
    df = df[df['text'].str.strip() != '']
    df['is_chapter'] = df.apply(is_chapter, axis=1) 
    df['is_section'] = df.apply(is_section, axis=1) 
    df['is_subsection'] = df.apply(is_subsection, axis=1) 
    df = df.drop(columns=['chars', 'fontsize', 'bolds', 'total', 'removable', 'page'])
    df['drop'] = False

    # The next loop checks if there are multiline chapters, sections or sub-sections
    # and merges them into one liners
    new_records = df.to_records()
    i = 0
    while i < len(new_records):
        if new_records[i]['is_chapter']:
            text = new_records[i]['text']
            j = i
            while j + 1 < len(new_records) and new_records[j + 1]['is_chapter']:
                j += 1
                new_records[j]['drop'] = True
                text += '\n' + new_records[j]['text']
            new_records[i]['text'] = text
            i = j + 1
        elif new_records[i]['is_section']:
            text = new_records[i]['text']
            j = i
            while j + 1 < len(new_records) and new_records[j + 1]['is_section']:
                j += 1
                new_records[j]['drop'] = True
                text += '\n' + new_records[j]['text']
            new_records[i]['text'] = text
            i = j + 1
        elif new_records[i]['is_subsection']:
            text = new_records[i]['text']
            j = i
            while j + 1 < len(new_records) and new_records[j + 1]['is_subsection']:
                j += 1
                new_records[j]['drop'] = True
                text += '\n' + new_records[j]['text']
            new_records[i]['text'] = text
            i = j + 1
        else:
            i += 1
    df = pd.DataFrame.from_records(new_records)
    df = df[~df['drop']]
    df['text'] = df.apply(lambda row: row['text'] + '.' if (row['is_chapter'] or row['is_section'] or row['is_subsection']) else row['text'] , axis=1)

    # index chapters, sections and subsections
    df['chapter_idx'] = df['is_chapter'].cumsum()
    df['section_idx'] = df.groupby(['chapter_idx'])['is_section'].cumsum()
    df['subsection_idx'] = df.groupby(['chapter_idx', 'section_idx'])['is_subsection'].cumsum()

    # extract the names of chapters, sections and subsections
    chapters = df.query("is_chapter == True")[['text','chapter_idx']].rename(columns={'text':'chapter'})
    sections = df.query("is_section == True")[['text','chapter_idx','section_idx']].rename(columns={'text':'section'})
    sub_sections = df.query("is_subsection == True")[['text','chapter_idx','section_idx', 'subsection_idx']].rename(columns={'text':'subsection'})

    # Merge the lines corresponding to the same chapter, section, subsection
    df = df.merge(chapters, on=['chapter_idx'], how='left')
    df = df.merge(sections, on=['chapter_idx','section_idx'], how='left')
    df = df.merge(sub_sections, on=['chapter_idx','section_idx', 'subsection_idx'], how='left')
    df.fillna('-', inplace=True)
    df = df.drop(columns=['drop', 'is_chapter', 'is_section', 'is_subsection', 'chapter_idx', 'section_idx', 'subsection_idx'])
    df = df.groupby(['chapter', 'section', 'subsection'], as_index=False, sort=False).agg(
        text = ('text', lambda x: '\n'.join(x)))
    df = df[df['chapter'] != df['text']]
    df = df[df['section'] != df['text']]
    df = df.reset_index(drop=True)
    df['id'] = df.apply(lambda row:
                        row['chapter'][:row['chapter'].find('.')] + 
                        ('' if  row['section'] == '-' else '.' + row['section'][:row['section'].find('.')]) + 
                        ('' if  row['subsection'] == '-' else '.' + row['subsection'][:row['subsection'].find(')')]), axis=1)
    return df
