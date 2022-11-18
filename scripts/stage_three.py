import os
import re
import json
import sys
import time
import nltk # natural language processing
import shutil # helper function for prepare_text_data() moves files from one folder to another
from colorama import Fore, Back, Style # for colored terminal output

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Aliased Libraries
import pandas as pd
import numpy as np

# global flags
print_flag = True # verbose printing flag

# sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from art import text2art # helper functions
from rich.progress import track
from time import sleep
from tqdm import tqdm

def build_data_viz():
    # this function will build the data visualization in holoviz using the data in the book_data folder
    # year,author,title,text,keywords,all_years,common_words,entities
    return

def clean_text_thoroughly(novel_text,regex_patterns_dict):

    original_length = len(novel_text)
    # this function will clean the text data in each text file
    #^ helper function for prepare_text_data()
    # this function will clean the text data in each of the text files in the data/firsthand/novels folder and save the cleaned text data to the data/firsthand/novels_clean folder as a text file. If this folder does not exist, it will be created.
    # novel_text is the string contents of a file that was opened in the prepare_text_data() function
    # step 1.1. remove the header and footer from the text data in each text file
    novel_text = re.sub(r'(?s)^\s*.*?START OF THIS PROJECT GUTENBERG EBOOK.*?\n', '', novel_text)
    novel_text = re.sub(r'(?s)END OF THIS PROJECT GUTENBERG EBOOK.*?\s*$', '', novel_text)
    # step 1.2. remove the chapter headers from the text data in each text file
    novel_text = re.sub(r'(?s)CHAPTER.*?\n', '', novel_text)
    print("> 1.2 ✔ ", end='')

    # preview the cleaned text data
        # print(f'Preview of the cleaned text data:')
    # print('---------------------------------------------------')
    # print(novel_text[:1000])
    # print('...')

    # combine all the chapter patterns into one pattern
    #master_chapter_pattern = r'(?s)^\s*CHAPTER.*?\n|^\s*\d+.*?\n|^\s*[IVXLC]+.*?\n|^\s*[ivxlc]+.*?\n|^\s*\d+\s+.*?\n|^\s*[IVXLC]+\s+.*?\n|^\s*[ivxlc]+\s+.*?\n|^\s*\d+\s+.*?\n|^\s*[IVXLC]+\s+.*?\n|^\s*[ivxlc]+\s+.*?\n|^\s*\d+\s+.*?\n' # add the rest of the chapter patterns here
    master_chapter_pattern = regex_patterns_dict['master_chapter_pattern'] # add the rest of the chapter patterns here
    # remove the chapter headers from the text data in each text file
    novel_text = re.sub(master_chapter_pattern, '', novel_text)
    # step 1.3. remove numbers from the text data in each text file
    number_pattern = regex_patterns_dict['number_pattern']
    # remove numbers from the text data in each text file
    novel_text = re.sub(number_pattern, '', novel_text)
    print("> 1.3 ✔ ", end='')

    # preview the cleaned text data
        # print(f'Preview of the cleaned text data:')
    # print('---------------------------------------------------')
    # print(novel_text[:1000])
    # print('...')


    # step 1.4. remove punctuation from the text data in each text file
    punctuation_pattern = regex_patterns_dict['punctuation_pattern']
    # remove punctuation from the text data in each text file
    novel_text = re.sub(punctuation_pattern, '', novel_text)
    print(f"> 1.4 ✔ {original_length-len(novel_text)} characters trimmed")
    # step 1.5. remove whitespace from the text data in each text file
    # whitespace_pattern = regex_patterns_dict['whitespace_pattern']
    # remove whitespace from the text data in each text file
    # novel_text = re.sub(whitespace_pattern, '', novel_text)
    # print("> XXX ", end='')

    # preview the cleaned text data
        # print(f'Preview of the cleaned text data:')
    # print('---------------------------------------------------')
    # print(novel_text[:1000])
    # print('...')

    # step 1.6. remove the Project Gutenberg header from the text data in each text file
    project_gutenberg_header_pattern = regex_patterns_dict['project_gutenberg_header_pattern']
    # remove the Project Gutenberg header from the text data in each text file
    novel_text = re.sub(project_gutenberg_header_pattern, '', novel_text)
    print(f"> 1.6 ✔ {original_length-len(novel_text)} characters trimmed")

    # preview the cleaned text data
        # print(f'Preview of the cleaned text data:')
    # print('---------------------------------------------------')
    # print(novel_text[:1000])
    # print('...')

    # step 1.7. remove the Project Gutenberg footer from the text data in each text file
    project_gutenberg_footer_pattern = regex_patterns_dict['project_gutenberg_footer_pattern']
    # remove the Project Gutenberg footer from the text data in each text file
    novel_text = re.sub(project_gutenberg_footer_pattern, '', novel_text)
    print(f"> 1.7 ✔ {original_length-len(novel_text)} characters trimmed")

    # preview the cleaned text data
        # print(f'Preview of the cleaned text data:')
    # print('---------------------------------------------------')
    # print(novel_text[:1000])
    # print('...')

    # step 1.8. remove the Project Gutenberg license from the text data in each text file
    project_gutenberg_license_pattern = regex_patterns_dict['project_gutenberg_license_pattern']
    # remove the Project Gutenberg license from the text data in each text file
    novel_text = re.sub(project_gutenberg_license_pattern, '', novel_text)
    print(f"> 1.8 ✔ {original_length-len(novel_text)} characters trimmed")

    # preview the cleaned text data
        # print(f'Preview of the cleaned text data:')
    # print('---------------------------------------------------')
    # print(novel_text[:1000])
    # print('...')

    # step 1.9. stopwords_pattern removal
    #!stopwords_pattern = regex_patterns_dict['stopwords_pattern']
    # remove stopwords from the text data in each text file
    #!novel_text = re.sub(stopwords_pattern, '', novel_text)
    print(f"> XXX ✔ {original_length-len(novel_text)} characters trimmed")

    # preview the cleaned text data
    # print(f'Preview of the cleaned text data:')
    # print('---------------------------------------------------')
    # print(novel_text[:1000])
    # print('...')

    # step 1.10. remove the nonenglish_pattern from the text data in each text file
    #nonenglish_pattern = regex_patterns_dict['nonenglish_pattern']
    # remove the nonenglish_pattern from the text data in each text file
    #novel_text = re.sub(nonenglish_pattern, '', novel_text)
    # remove double \n\n from the text data in each text file
    #!print(f'Removing double newlines from the text data in each text file')
    # novel_text = re.sub(r'(?s)\n\n', '\n', novel_text)
    # print("> 1.10 ✔ ", end='')


    # preview the cleaned text data
    # print(f'Preview of the cleaned text data:')
    # print('---------------------------------------------------')
    # print(novel_text[:1000])
    # print('...')

    # step 1.11. remove any double spaces from the text data in each text file
    novel_text = re.sub(r'(?s)  ', ' ', novel_text)
    print(f"> 1.10 {original_length-len(novel_text)} characters trimmed")
    if '  ' not in novel_text:
        print("> ✅ ", end='')
    else:
        while '  ' in novel_text:
            novel_text = re.sub(r'(?s)  ', ' ', novel_text)
        print(f"> ✅ {original_length-len(novel_text)} characters trimmed")
    # step 1.12. remove any double newlines from the text data in each text file
    novel_text = re.sub(r'(?s)\n\n', '\n', novel_text)
    if '\n\n' not in novel_text:
        print(f"> 1.11  {original_length-len(novel_text)} characters trimmed")
    else:
        print('\n|', end='')
        while '\n\n' in novel_text:
            print('😒', end='')
            novel_text = re.sub(r'(?s)\n\n', '\n', novel_text)

        print(f"\n> 1.12 {original_length-len(novel_text)} characters trimmed")

    # preview the cleaned text data
        # print(f'Preview of the cleaned text data:')
    # print('---------------------------------------------------')
    # print(novel_text[:1000])
    # print('...')


    print("> XXX --> DONE")
    return novel_text

def clean_text_keep_punct(novel_text,regex_patterns_dict):
    global print_flag
    # silence print statements if boolean print_flag is False
    if print_flag == False:
        sys.stdout = open(os.devnull, 'w')

    original_length = len(novel_text)
    # this function will clean the text data in each text file
    #^ helper function for prepare_text_data()
    # this function will clean the text data in each of the text files in the data/firsthand/novels folder and save the cleaned text data to the data/firsthand/novels_clean folder as a text file. If this folder does not exist, it will be created.
    # novel_text is the string contents of a file that was opened in the prepare_text_data() function
    # step 1.1. remove the header and footer from the text data in each text file
    novel_text = re.sub(r'(?s)^\s*.*?START OF THIS PROJECT GUTENBERG EBOOK.*?\n', '', novel_text)
    novel_text = re.sub(r'(?s)END OF THIS PROJECT GUTENBERG EBOOK.*?\s*$', '', novel_text)
    # step 1.2. remove the chapter headers from the text data in each text file
    novel_text = re.sub(r'(?s)CHAPTER.*?\n', '', novel_text)
    print("> 1.2 ✔ ", end='')

    # preview the cleaned text data
        # print(f'Preview of the cleaned text data:')
    # print('---------------------------------------------------')
    # print(novel_text[:1000])
    # print('...')

    # combine all the chapter patterns into one pattern
    #master_chapter_pattern = r'(?s)^\s*CHAPTER.*?\n|^\s*\d+.*?\n|^\s*[IVXLC]+.*?\n|^\s*[ivxlc]+.*?\n|^\s*\d+\s+.*?\n|^\s*[IVXLC]+\s+.*?\n|^\s*[ivxlc]+\s+.*?\n|^\s*\d+\s+.*?\n|^\s*[IVXLC]+\s+.*?\n|^\s*[ivxlc]+\s+.*?\n|^\s*\d+\s+.*?\n' # add the rest of the chapter patterns here
    master_chapter_pattern = regex_patterns_dict['master_chapter_pattern'] # add the rest of the chapter patterns here
    # remove the chapter headers from the text data in each text file
    novel_text = re.sub(master_chapter_pattern, '', novel_text)
    # step 1.3. remove numbers from the text data in each text file
    number_pattern = regex_patterns_dict['number_pattern']
    # remove numbers from the text data in each text file
    novel_text = re.sub(number_pattern, '', novel_text)
    # print "> 1.3 ✔ " with colorama green if the number_pattern was found in the text data, otherwise print with colorama blue
    if re.search(number_pattern, novel_text) is not None: # if the number_pattern was found in the text data
        print(Fore.GREEN + f"> 1.3 ✔ {original_length-len(novel_text)} characters trimmed" + Fore.RESET)
    else:
        print(Fore.BLUE + f"> 1.3 ✔ {original_length-len(novel_text)} characters trimmed" + Fore.RESET)

    # step 1.6. remove the Project Gutenberg header from the text data in each text file
    project_gutenberg_header_pattern = regex_patterns_dict['project_gutenberg_header_pattern']
    # remove the Project Gutenberg header from the text data in each text file
    novel_text = re.sub(project_gutenberg_header_pattern, '', novel_text)
    if re.search(number_pattern, novel_text) is not None: # if the number_pattern was found in the text data
        print(Fore.GREEN + f"> 1.6 ✔ {original_length-len(novel_text)} characters trimmed" + Fore.RESET)
    else:
        print(Fore.BLUE + f"> 1.6 ✔ {original_length-len(novel_text)} characters trimmed" + Fore.RESET)

    # step 1.7. remove the Project Gutenberg footer from the text data in each text file
    project_gutenberg_footer_pattern = regex_patterns_dict['project_gutenberg_footer_pattern']
    # remove the Project Gutenberg footer from the text data in each text file
    novel_text = re.sub(project_gutenberg_footer_pattern, '', novel_text)
    if re.search(number_pattern, novel_text) is not None: # if the number_pattern was found in the text data
        print(Fore.GREEN + f"> 1.7 ✔ {original_length-len(novel_text)} characters trimmed" + Fore.RESET)
    else:
        print(Fore.BLUE + f"> 1.7 ✔ {original_length-len(novel_text)} characters trimmed" + Fore.RESET)

    # step 1.8. remove the Project Gutenberg license from the text data in each text file
    project_gutenberg_license_pattern = regex_patterns_dict['project_gutenberg_license_pattern']
    # remove the Project Gutenberg license from the text data in each text file
    novel_text = re.sub(project_gutenberg_license_pattern, '', novel_text)
    if re.search(number_pattern, novel_text) is not None: # if the number_pattern was found in the text data
        print(Fore.GREEN + f"> 1.8 ✔ {original_length-len(novel_text)} characters trimmed" + Fore.RESET)
    else:
        print(Fore.BLUE + f"> 1.8 ✔ {original_length-len(novel_text)} characters trimmed" + Fore.RESET)

    # step 1.11. remove any double spaces from the text data in each text file
    novel_text = re.sub(r'(?s)  ', ' ', novel_text)
    if re.search(number_pattern, novel_text) is not None: # if the number_pattern was found in the text data
        print(Fore.GREEN + f"> 1.9 ✔ {original_length-len(novel_text)} characters trimmed" + Fore.RESET)
    else:
        print(Fore.BLUE + f"> 1.9 ✔ {original_length-len(novel_text)} characters trimmed" + Fore.RESET)
    if '  ' not in novel_text:
        print("> ✅ ", end='')
    else:
        while '  ' in novel_text:
            novel_text = re.sub(r'(?s)  ', ' ', novel_text)
        print(f"> ✅ {original_length-len(novel_text)} characters trimmed")
    # step 1.12. remove any double newlines from the text data in each text file
    novel_text = re.sub(r'(?s)\n\n', '\n', novel_text)
    if '\n\n' not in novel_text:
        if re.search(number_pattern, novel_text) is not None: # if the number_pattern was found in the text data
            print(Fore.GREEN + f"> 1.10 ✔ {original_length-len(novel_text)} characters trimmed" + Fore.RESET)
        else:
            print(Fore.BLUE + f"> 1.10 ✔ {original_length-len(novel_text)} characters trimmed" + Fore.RESET)
    else:
        print('\n|', end='')
        while '\n\n' in novel_text:
            print('😒', end='')
            novel_text = re.sub(r'(?s)\n\n', '\n', novel_text)

        if re.search(number_pattern, novel_text) is not None: # if the number_pattern was found in the text data
            print(Fore.GREEN + f"> 1.11 ✔ {original_length-len(novel_text)} characters trimmed" + Fore.RESET)
        else:
            print(Fore.BLUE + f"> 1.11 ✔ {original_length-len(novel_text)} characters trimmed" + Fore.RESET)


    print(Fore.YELLOW + "> XXX --> DONE" + Fore.RESET) # print the end of the function with colorama yellow
    return novel_text

def prepare_text_data():
    # this function will call its helper functions to prepare the data for modeling in each of the text files in the data/firsthand/novels folder and save the cleaned text data to the data/firsthand/novels_clean folder as a text file. If this folder does not exist, it will be created.
    # read the regex patterns from the json file
    with open('./config/regex_patterns.json', 'r') as file_reading:
        regex_patterns_dict = json.load(file_reading)
    print('Successfully loaded the regex patterns from the json file.')

    # step 0. open each text file in the data/firsthand/novels folder
    for each_novel in tqdm(os.listdir('./data/firsthand/novels')):
        # open the file and read the text into a string variable `novel_text`
        with open('./data/firsthand/novels/' + each_novel, 'r') as file_reading:
            novel_text = file_reading.read()

        # step 1. clean the text data in each text file
        print(f'Step 1. Cleaning text data in {each_novel}...')
        novel_text = clean_text_keep_punct(novel_text, regex_patterns_dict) # call the clean_text() function (keep punctuation) to clean the text data in each text file.
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

def prep_data_for_cvec():
    # prep all the text files in the data/firsthand/novels_clean folder for CountVectorizer
    # make a copy of the folder into another folder called data/firsthand/novels_clean_cvec to preserve the original data
    # create the data/firsthand/novels_clean_cvec folder if it does not exist
    if not os.path.exists('./data/firsthand/novels_clean_cvec'):
        os.makedirs('./data/firsthand/novels_clean_cvec')
    # copy the data/firsthand/novels_clean folder into the data/firsthand/novels_clean_cvec folder
    for each_novel in tqdm(os.listdir('./data/firsthand/novels_clean')):
        shutil.copyfile('./data/firsthand/novels_clean/' + each_novel, './data/firsthand/novels_clean_cvec/' + each_novel)


def count_dracula():
    # implement sklearn CountVectorizer to count the number of times each word appears in each text file in the data/firsthand/novels_clean_cvec folder. The results will be saved to the data/cvec folder as a csv file with the name of the text file as the file name. If this folder does not exist, it will be created and the csv file will be saved there.

    # step 0. create the data/cvec folder if it does not exist
    if not os.path.exists('./data/cvec'):
        os.makedirs('./data/cvec')
    # step 1. implement sklearn CountVectorizer to count the number of times each word appears in each text file in the data/firsthand/novels_clean_cvec folder
    # create a list of all the text files in the data/firsthand/novels_clean_cvec folder
    novels_list = os.listdir('./data/firsthand/novels_clean_cvec')


    # print a nice styled title saying "CVEC" for this section that is made of ascii characters and uses colorama to color the text. The title is centered using the center() method."

    # print a nice styled title saying "CVEC" for this section that is made of ascii characters and uses colorama to color the text. The title is centered using the center() method."
    print(Fore.CYAN + Style.BRIGHT + f"{'Initializing Count Vectorizer':^80}".center(80, '#') + Style.RESET_ALL + Fore.RESET)

    # use text2art to make a banner saying "CVEC" (green font) and print it
    Art = text2art("CVEC", font="block", chr_ignore=True).center(80, ' ')
    print(Fore.GREEN + Style.BRIGHT + Art + Style.RESET_ALL + Fore.RESET)

    time.sleep(1)
    print(Fore.GREEN + Style.BRIGHT + f"{'Let the count begin...':^80}".center(80, '#') + Style.RESET_ALL + Fore.RESET)
    for novel_i in tqdm(novels_list): # loop through each text file in the data/firsthand/novels_clean_cvec folder
        # open the text file and read the text into a string variable `novel_text`
        try:
            with open('./data/firsthand/novels_clean_cvec/' + novel_i, 'r') as file_reading:
                novel_text = file_reading.read()
            # implement sklearn CountVectorizer to count the number of times each word appears in each text file in the data/firsthand/novels_clean_cvec folder
            cvec = CountVectorizer()
            cvec.fit([novel_text])
            # save the results to the data/cvec folder as a csv file with the name of the text file as the file name
            # use get_feature_names_out() to get the list of all the words in the text file instead of get_feature_names() to get the list of all the words in the corpus. This will be deprecated in the future.
            cvec_df = pd.DataFrame(cvec.transform([novel_text]).todense(), columns=cvec.get_feature_names_out(), index=[novel_i])
            cvec_df.to_csv('./data/cvec/ ' + novel_i + '_cvec.csv') # save the results to the data/cvec folder as a csv file with the name of the text file as the file name
            # print the name of the book and the number of words in the book to the console. Use colorama to print the name of the book in yellow and the number of words in the book in green. Use the end='' argument to print the name of the book and the number of words in the book on the same line.
            print(Fore.YELLOW + novel_i + Fore.RESET + " has been read into the system. ", end='')
            print(Fore.GREEN + f' has {cvec_df.sum().sum()} words.' + Fore.RESET)
        except Exception as e:
            print(Fore.RED + f'Failed to save the results to the data/cvec folder as a csv file with the name of the text file as the file name. Error: {e}' + Fore.RESET)




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
    print(Fore.CYAN + Style.BRIGHT + f"{'Preparing the text data'}" + Style.RESET_ALL + Fore.RESET)
    print('-'*len('Preparing the text data'))
    time.sleep(1)
    prepare_text_data() # > calls the prepare_text_data() function > calls the clean_text_keep_punct() function which saves each novel as a text file in the data/firsthand/novels_clean folder and maintains the punctuation in the text data.
    print(Fore.GREEN + Style.BRIGHT + f"{'Text data has been prepared and saved to the data/firsthand/novels_clean folder':^80}".center(80, '#') + Style.RESET_ALL + Fore.RESET)
    prep_data_for_cvec() # > calls the prep_data_for_cvec() function which makes a copy of the data/firsthand/novels_clean folder into the data/firsthand/novels_clean_cvec folder to preserve the original data.
    #* count vectorize the data/firsthand/novels_clean_cvec folder and save the vectorized data to the data/firsthand/novels_clean_cvec folder as a csv file. If this folder does not exist, it will be created.
    count_dracula()
    return


def save_patterns_as_json(regex_patterns_dict):
    # this function will save the regex patterns as a json file
    with open('regex_patterns.json', 'w') as file_writing:
        json.dump(regex_patterns_dict, file_writing)
    return


if __name__ == "__main__":
    main()