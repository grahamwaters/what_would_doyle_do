import pandas as pd
import numpy as np
import os
import sys
import re
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
# sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


def build_data_viz():
    # this function will build the data visualization in holoviz using the data in the book_data folder
    # year,author,title,text,keywords,all_years,common_words,entities
    return






def clean_text(novel_text,regex_patterns_dict):
    # this function will clean the text data in each text file
    #^ helper function for prepare_text_data()
    # this function will clean the text data in each of the text files in the data/firsthand/novels folder and save the cleaned text data to the data/firsthand/novels_clean folder as a text file. If this folder does not exist, it will be created.
    # novel_text is the string contents of a file that was opened in the prepare_text_data() function
    # step 1.1. remove the header and footer from the text data in each text file
    novel_text = re.sub(r'(?s)^\s*.*?START OF THIS PROJECT GUTENBERG EBOOK.*?\n', '', novel_text)
    novel_text = re.sub(r'(?s)END OF THIS PROJECT GUTENBERG EBOOK.*?\s*$', '', novel_text)
    # step 1.2. remove the chapter headers from the text data in each text file
    novel_text = re.sub(r'(?s)CHAPTER.*?\n', '', novel_text)
    print("> 1.2 ", end='')
    # combine all the chapter patterns into one pattern
    #master_chapter_pattern = r'(?s)^\s*CHAPTER.*?\n|^\s*\d+.*?\n|^\s*[IVXLC]+.*?\n|^\s*[ivxlc]+.*?\n|^\s*\d+\s+.*?\n|^\s*[IVXLC]+\s+.*?\n|^\s*[ivxlc]+\s+.*?\n|^\s*\d+\s+.*?\n|^\s*[IVXLC]+\s+.*?\n|^\s*[ivxlc]+\s+.*?\n|^\s*\d+\s+.*?\n' # add the rest of the chapter patterns here
    master_chapter_pattern = regex_patterns_dict['master_chapter_pattern'] # add the rest of the chapter patterns here
    # remove the chapter headers from the text data in each text file
    novel_text = re.sub(master_chapter_pattern, '', novel_text)
    # step 1.3. remove numbers from the text data in each text file
    number_pattern = regex_patterns_dict['number_pattern']
    # remove numbers from the text data in each text file
    print("> 1.3 ", end='')
    novel_text = re.sub(number_pattern, '', novel_text)
    # step 1.4. remove punctuation from the text data in each text file
    punctuation_pattern = regex_patterns_dict['punctuation_pattern']
    # remove punctuation from the text data in each text file
    novel_text = re.sub(punctuation_pattern, '', novel_text)
    print("> 1.4 ", end='')
    # step 1.5. remove whitespace from the text data in each text file
    # whitespace_pattern = regex_patterns_dict['whitespace_pattern']
    # remove whitespace from the text data in each text file
    # novel_text = re.sub(whitespace_pattern, '', novel_text)
    print("> XXX ", end='')
    # step 1.6. remove the Project Gutenberg header from the text data in each text file
    project_gutenberg_header_pattern = regex_patterns_dict['project_gutenberg_header_pattern']
    # remove the Project Gutenberg header from the text data in each text file
    novel_text = re.sub(project_gutenberg_header_pattern, '', novel_text)
    print("> 1.6 ", end='')
    # step 1.7. remove the Project Gutenberg footer from the text data in each text file
    project_gutenberg_footer_pattern = regex_patterns_dict['project_gutenberg_footer_pattern']
    # remove the Project Gutenberg footer from the text data in each text file
    novel_text = re.sub(project_gutenberg_footer_pattern, '', novel_text)
    print("> 1.7 ", end='')
    # step 1.8. remove the Project Gutenberg license from the text data in each text file
    project_gutenberg_license_pattern = regex_patterns_dict['project_gutenberg_license_pattern']
    # remove the Project Gutenberg license from the text data in each text file
    novel_text = re.sub(project_gutenberg_license_pattern, '', novel_text)
    print("> 1.8 ", end='')
    # step 1.9. stopwords_pattern removal
    stopwords_pattern = regex_patterns_dict['stopwords_pattern']
    # remove stopwords from the text data in each text file
    novel_text = re.sub(stopwords_pattern, '', novel_text)
    print("> 1.9 ", end='')
    # step 1.10. remove the nonenglish_pattern from the text data in each text file
    nonenglish_pattern = regex_patterns_dict['nonenglish_pattern']
    # remove the nonenglish_pattern from the text data in each text file
    novel_text = re.sub(nonenglish_pattern, '', novel_text)
    print("> 1.10 --> DONE")
    return novel_text



def prepare_text_data():
    # this function will call its helper functions to prepare the data for modeling in each of the text files in the data/firsthand/novels folder and save the cleaned text data to the data/firsthand/novels_clean folder as a text file. If this folder does not exist, it will be created.
    # read the regex patterns from the json file
    with open('./config/regex_patterns.json', 'r') as file_reading:
        regex_patterns_dict = json.load(file_reading)
    print('Successfully loaded the regex patterns from the json file.')

    # step 0. open each text file in the data/firsthand/novels folder
    for each_novel in os.listdir('./data/firsthand/novels'):
        # open the file and read the text into a string variable `novel_text`
        with open('./data/firsthand/novels/' + each_novel, 'r') as file_reading:
            novel_text = file_reading.read()

        # step 1. clean the text data in each text file
        print(f'Step 1. Cleaning text data in {each_novel}...')
        novel_text = clean_text(novel_text, regex_patterns_dict) # call the clean_text() function
        # step 2. save the cleaned text data to the data/firsthand/novels_clean folder as a text file
        print(f'Step 2. Saving cleaned text ...')
        # create the data/firsthand/novels_clean folder if it does not exist
        if not os.path.exists('./data/firsthand/novels_clean'):
            os.makedirs('./data/firsthand/novels_clean')
        # save the cleaned text data to the data/firsthand/novels_clean folder as a text file
        with open('./data/firsthand/novels_clean/' + each_novel, 'w') as file_writing:
            file_writing.write(novel_text)
        print(f'Successfully saved the cleaned text data to the data/firsthand/novels_clean folder as a text file.')
    print('Successfully prepared the text data for modeling in each of the text files in the data/firsthand/novels folder and saved the cleaned text data to the data/firsthand/novels_clean folder as a text file.')


    return


def main():
    # regex_patterns_dict = {
    # 'master_chapter_pattern':r'(?s)^\s*CHAPTER.*?\n|^\s*\d+.*?\n|^\s*[IVXLC]+.*?\n|^\s*[ivxlc]+.*?\n|^\s*\d+\s+.*?\n|^\s*[IVXLC]+\s+.*?\n|^\s*[ivxlc]+\s+.*?\n|^\s*\d+\s+.*?\n|^\s*[IVXLC]+\s+.*?\n|^\s*[ivxlc]+\s+.*?\n|^\s*\d+\s+.*?\n',
    # 'number_pattern':r'\d+',
    # 'punctuation_pattern':r'[^\w\s]',
    # 'whitespace_pattern':r'\s+',
    # 'stopwords_pattern':r'\b(' + r'|'.join(nltk.corpus.stopwords.words('english')) + r')\b\s*',
    # 'nonenglish_pattern':r'\b(' + r'|'.join(nltk.corpus.words.words()) + r')\b\s*',
    # 'project_gutenberg_header_pattern':r'(?s)^\s*Project Gutenberg.*?\n',
    # 'project_gutenberg_footer_pattern':r'(?s)^\s*End of the Project Gutenberg.*?\n',
    # 'project_gutenberg_license_pattern':r'(?s)^\s*This eBook is for the use of.*?\n'
    # }
    #save_patterns_as_json(regex_patterns_dict)


    # step 0. prepare the text data in each of the text files in the data/firsthand/novels folder and save the cleaned text data to the data/firsthand/novels_clean folder as a text file. If this folder does not exist, it will be created.
    prepare_text_data() # > calls the prepare_text_data() function > calls the clean_text() function > calls the tokenize_text() function > calls the lemmatize_text() function > calls the create_ngrams() function > calls the create_bigrams() function > calls the create_trigrams() function
    # step 1. build the data visualization in holoviz using the data in the book_data folder
    # build_data_viz() # > calls the build_data_viz() function

    return

import json
def save_patterns_as_json(regex_patterns_dict):
    # this function will save the regex patterns as a json file
    with open('regex_patterns.json', 'w') as file_writing:
        json.dump(regex_patterns_dict, file_writing)
    return


if __name__ == "__main__":
    main()