# Stage Two of the Data Science Process: Data Gathering
"""
    Script Summary:
    What data can we collect about Arthur Conan Doyle that will enable us to solve this problem? We need every book he's ever written, which is around 80 books and is provided through the Gutenberg repository. These books are included in the Data folder as text files; second, I would like to have anything he wrote that was a first-hand account because this is where we will get his personal preferences and his turns of phrase, and maybe even his personal biases, which are probably the most important things to gather once we gather his diaries, journals. Things other people said about him are the next step. Now we want to gather any second-hand accounts of Doyle. Many people have done years of research on historical figures and repeating it seems like a useless task and honestly is a waste of precious resources. So in this step, we want to gather any biographies that were written about Arthur Conan Doyle and any articles that were written about him, especially if they were written about him in the time that he lived. And this might be most useful if we were to gather the names of all of his second-degree connections. If we think about it, in terms of a LinkedIn network, though, Doyle's second-degree connections are the ones that are most likely to have the most accurate depictions of his preferences. This is, of course, an assumption that I am making. Once we gather the names of his second-degree connections, I think it would be an excellent step to assign weight to their accounts if they authored anything.

"""
import time
import csv
import os
import pandas
import pickle
import spacy # for NLP
nlp = spacy.load('en_core_web_md')



def book_title_reformatter(book_title):
    # remove duplicated words from the title string
    for word in book_title.split("_"):
        # find duplicated pairs of words with a number first then the word doyle
        if book_title.count(word) > 1:
            book_title = book_title.replace(f'_{word}','')
    # last check to make sure the title is formatted correctly
    if re.search(r'\d{1,2}.*doyle',word.lower()):
        book_title = book_title.replace(word,'')
    # add doyle back in between the numbers and the title
    book_title = str(re.sub(r'(\d{1,4})(.*)',r'\1_doyle_\2',book_title))\
        .replace('__','_')\
        .replace('doyle_doyle','doyle')
    # remove __ by replacing it with _
    # book_title = book_title.replace('__','_')
    #^ replace instances of 'doyle_doyle' with 'doyle'
    # book_title = book_title.replace('doyle_doyle','doyle')
    return book_title


def bookshelf_mass_rename(books_folder):
    # for every book in the books folder, rename the books with book_title_reformatter
    for root, dirs, files in os.walk(books_folder):
        for file in files:
            # rename the file
            os.rename(f'{root}/{file}',f'{root}/{book_title_reformatter(file)}')
            #* remove any ds_store files that may be in the directory (using regex)
            #*remove_ds_store(root,file)




# using SpaCy to get keywords
def get_keywords(book_title,book_text,bar):

    #^ use reformatter
    book_title = book_title_reformatter(book_title) # remove any duplicated words from the title


    # check if the pickled file exists already
    if os.path.exists(f'./data/pickles/{book_title}.pkl'):
        print(f'Found pickled file for {book_title}')
        return

    # using SpaCy to get keywords
    try:
        doc = nlp(book_text)
        docs = [doc]
    except ValueError: # the text is too large
        # split the text into chunks
        doc2 = nlp(book_text[:len(book_text)//2])
        doc1 = nlp(book_text[len(book_text)//2:])
        docs = [doc1, doc2]
        #note: you may need to extend this to more chunks if the text is still too large, but this should be sufficient for most cases
    all_keys = []
    for doc in docs:
        if len(docs)>1: print("Chunking text...")
        keywords = [token.lemma_ for token in doc if token.pos_ == 'NOUN' and token.is_stop == False] # get all the nouns that are not stopwords and lemmatize them

        # while we're at it, let's get the most common words, too and the entities (people, places, things)
        bar.text('Getting most common words...')
        common_words = [token.text for token in doc if token.is_stop == False and token.is_punct == False and token.pos_ != 'NOUN']
        bar.text(f'Getting entities...')
        entities = [ent.text for ent in doc.ents]

        # remove duplicates
        keywords = list(set(keywords))
        all_keys.extend(keywords) # add the keywords to the list
        common_words = list(set(common_words))
        entities = list(set(entities))

        # save these to files for later use (pickle)
        # remove .txt from the title
        title = book_title.replace('.txt','')

        # save the keywords to a pickle file for later use (this will be used in the next stage)
        # if the folder doesn't exist, create it first
        if not os.path.exists('./data/pickles'):
            os.mkdir('./data/pickles')
        if not os.path.exists(f'./data/pickles/{title}') and len(title)>0 and title != ' ':
            os.mkdir(f'./data/pickles/{title}') # create a folder for the book
        # save the keywords
        with open(f'./data/pickles/{title}/keywords.pkl','wb') as f:
            pickle.dump(keywords,f) # save the keywords
            # pickle.dump(','.join(keywords),f)
        # save the common words
        with open(f'./data/pickles/{title}/common_words.pkl','wb') as f:
            pickle.dump(common_words,f)
        # save the entities
        with open(f'./data/pickles/{title}/entities.pkl','wb') as f:
            pickle.dump(entities,f)

    return all_keys, common_words, entities




# All our firsthand, secondhand, and extended accounts are provided in the /data directory.
#? books_by_doyle = 'data/firsthand/novels'
#? journals_by_doyle = 'data/firsthand/journals'
#? articles_by_doyle = 'data/firsthand/articles'
#& books_about_doyle = 'data/secondhand/novels' # These are biographies
#& articles_about_doyle = 'data/secondhand/articles'
#& journals_mentioning_doyle = 'data/secondhand/journals' # If their journals mention him, we want to gather the relevant chapters from these journals to include in our knowledge base.
#* extended_accounts = 'data/extended/journals' # These are the second-degree connections of Doyle and their journals, diaries, etc.
#* extended_accounts = 'data/extended/articles' # These are the second-degree connections of Doyle and their articles, biographies, if those exist and mention either Doyle, or a significant location in his life within the right time period.
#* extended_accounts = 'data/extended/novels' # These are the second-degree connections of Doyle and their novels, if those exist and mention either Doyle, or a significant location in his life within the right time period.

# first we need to go through the books_by_doyle directory and rename our books to be in a machine learning ready format.

# the format: year_author_title.txt (e.g. 1890_doyle_the_sign_of_four.txt)
# this will make it easier to sort our books by year and author, and will make it easier to parse the title of the book for keywords.

# we will also need to create a dataframe that contains the following information:
# year, author, title, text, and keywords
import re
from alive_progress import alive_bar
import os

def remove_ds_store(root,file):
    # remove any ds_store files that may be in the directory (using regex)
    file = file.upper() # make the file name uppercase
    if re.search(r'\.DS_Store',file):
        os.remove(os.path.join(root,file))
        print(f'Removed .DS_Store file')



def book_reformatter(books_directory):

    files = os.listdir(books_directory)
    if 'ds_store.txt' in files:
        try:
            # os remove
            os.remove(os.path.join(books_directory,'ds_store.txt'))
        except ValueError:
            books_directory = books_directory[1:] # remove the first element (ds_store.txt)


    with alive_bar(len(os.listdir(books_directory))) as bar:
        # change titles to be ML ready and in the format: year_author_title.txt
        for book in os.listdir(books_directory):
            if '.ds_store' in book.lower():
                bar()
                continue # skip the .DS_Store file
            # currently we have books with spaces in the title and disorganized capitalization.
            # we want to remove the spaces and make the capitalization consistent while leaving the .txt extension.
            # also, if the book already has a pickle folder for it, we don't need to reprocess it.
            # if the book has already been processed, we can skip it.
            if os.path.exists(f'./data/pickles/{book.replace(".txt","")}'):
                bar.text(f'Already processed {book}...')
                bar()
                continue
            original_book_name = book # save the original name of the book
            # if the original name already has the correct format, we don't need to change it
            if re.search(r'\d{4}_[a-z]+_[a-z]+\.txt',book):
                bar()
                continue
            # replace spaces with underscores
            book = book.replace(' ', '_')
            # remove the .txt extension
            book = book.replace('.txt', '')
            # make the title all lowercase
            book = book.lower()
            # remove non alphanumeric characters
            book = ''.join(e for e in book if e.isalnum() or e == '_')
            # remove double underscores
            book = book.replace('__', '_')
            # add the .txt extension back
            book = book + '.txt'
            # rename the file
            # get the year the book was published from the text of the file using regex.

            # if the name did not change, then we don't need to do anything else.
            if original_book_name == book:
                bar.text("Book has already been reformatted.")
                bar.text("uncomment line 100-101 to pass the files that have already been reformatted.")
                # bar()
                #continue


            # update the file name

            os.rename(f'{books_directory}/{original_book_name}', f'{books_directory}/{book}')

            #^ There are some books that do not have a year in them, so we will need to handle that.
            all_years = [] # this will be a list of all the years in the book
            with open(os.path.join(books_directory, book), 'r') as f:
                text = f.read()
                year = re.search(r'\d{4}', text)
                all_years = re.findall(r'[ ,]\d{4}[, ]', text) # this will find all the years in the book, years being 4 digits long and surrounded by spaces or commas.
                if year:
                    year = year.group() # get the year as a string
                    #.group() returns the first match of the regex search
                    bar.text(f'found: {year}')
                else:
                    year = 'unk'
                bar.text('Getting Keywords with SpaCy')
                #  remove the \n characters from the text
                text = text.replace('\n', ' ')
                # get the keywords from the text
                print(f'Detecting Keywords in {book}')
                keywords, common_words, entities = get_keywords(book,text,bar)

            #* data cleaning for all_years
            # strip them and convert them to integers do not remove duplicates, remove commas if they exist
            all_years = [int(x.strip().replace(',','')) for x in all_years]

            entities = [x.lower() for x in entities] # make them all lowercase
            #* data cleaning for entities
            # remove any extra spaces
            entities = [x.strip() for x in entities]
            entities = [x.replace('  ', ' ') for x in entities] # remove double spaces
            # replace numbers with their word equivalent
            entities = [x.replace(' 1 ', ' one ') for x in entities]
            entities = [x.replace(' 2 ', ' two ') for x in entities]
            entities = [x.replace(' 3 ', ' three ') for x in entities]
            entities = [x.replace(' 4 ', ' four ') for x in entities]
            entities = [x.replace(' 5 ', ' five ') for x in entities]
            entities = [x.replace(' 6 ', ' six ') for x in entities]
            entities = [x.replace(' 7 ', ' seven ') for x in entities]
            entities = [x.replace(' 8 ', ' eight ') for x in entities]
            entities = [x.replace(' 9 ', ' nine ') for x in entities]
            entities = [x.replace(' 0 ', ' zero ') for x in entities] #note: repl with map function and a dictionary
            # remove duplicates
            entities = list(set(entities))

            #* data cleaning for common_words
            # remove any extra spaces
            common_words = [x if x != ' ' else x.strip() for x in common_words]
            common_words = [x if x != ' ' else x.replace('  ', ' ') for x in common_words] # remove double spaces
            # if there are entries in common_words that contain only spaces or are empty, remove them
            common_words = [x for x in common_words if x != ' ' and x != '']
            print(f'Identified {len(keywords)} keywords, {len(entities)} entities, and {len(common_words)} common words in {book}')



            # if all of book, year, and book_directory are in os.path.join(book_directory,book), then we can use the book name as the title
            if all([\
                book in os.path.join(books_directory,book), \
                year in os.path.join(books_directory,book), \
                books_directory in os.path.join(books_directory,book)]):
                pass # do nothing
            else:
                os.rename(os.path.join(books_directory, book), \
                    os.path.join(books_directory, year + '_doyle_' + book)) # rename the file to the correct format
                # the lines above do the following:
                # os.rename(os.path.join(books_directory, book) - this is the original file name and path
                # os.path.join(books_directory, year + '_doyle_' + book) - this is the new file name and path. It is the year the book was published, the author's last name, and the book title.
                # but what if the original file already has the correct format? We don't want to rename it.
                # we can check if the original file name is the same as the new file name. If they are the same, then we don't need to rename the file.





            # create a dataframe with the following columns: year, author, title, text, and keywords
            # we will use the year, author, and title to sort our books by year and author, and we will use the text to create our knowledge base, and we will use the keywords to search for keywords in our knowledge base.




            #df_book = pandas.DataFrame(columns=['year', 'author', 'title', 'all_years','text', 'keywords'])
            #df_book['year'] = year # add the year to the dataframe
            #df_book['author'] = 'doyle'
            if '.txt' in book: # remove the .txt extension from the title
                book = book.replace('.txt', '')
            else:
                pass # add the title to the dataframe
            #df_book['all_years'] = all_years # this will be a list of all the years in the book
            #df_book['text'] = text # this is the text of the book
            # remove everything from the keywords in df_book
            #df_book['keywords'] = ['' for x in range(len(keywords))]
            # df_book['keywords'] = [key for key in keywords] # from SpaCy


            dict_book = {'year': year, 'author': 'doyle', 'title': book, 'text': text, 'keywords': keywords, 'all_years': all_years, 'common_words': common_words, 'entities': entities}


            # save the dictionary as a csv in a new folder under the data directory called 'book_data' and name the file exactly the same as the book was renamed to in the previous step.
            # not using pandas because it is easier to save a dictionary as a csv than a dataframe.
            # create the file if it does not exist

            with open(f'data/book_data/{year}_doyle_{book}.csv', 'w+') as f:
                w = csv.DictWriter(f, dict_book.keys())
                w.writeheader()
                w.writerow(dict_book)

            bar()









def main():
    print(f'-------------------------')
    print(f'  What Would Doyle Do?')
    print(f'-------------------------')
    time.sleep(0.25)
    print(f"Let's find out!")

    print('-----------------------------------')
    print(' Verfiying book naming conventions ')
    bookshelf_mass_rename('./data/book_data')
    bookshelf_mass_rename('./data/firsthand')
    print('-----------------------------------')
    time.sleep(0.25)

    try:
        book_reformatter('data/firsthand/novels')
    except Exception as e:
        print(f'Error: {e}')

    print('-----------------------------------')
    print('  All books are pickled and ready  ')
    print('-----------------------------------')

    time.sleep(0.25)






main()