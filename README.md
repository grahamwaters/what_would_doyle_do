

<!-- adding shields for contributor count/chat on discord and coverage -->
<h1 align="center" font-size = 40px;> What Would Doyle Do?
</h1>
<h2 align="center" font-size = 20px;> (W.W.D.D.)
</h2>

<div align="center">

[![contributions - welcome](https://img.shields.io/badge/contributions-welcome-blue)](/CONTRIBUTING.md "Go to contributions doc") [![Made with Python](https://img.shields.io/badge/Python->=3.6-blue?logo=python&logoColor=white)](https://python.org "Go to Python homepage")

![license](https://img.shields.io/github/license/grahamwaters/what_would_doyle_do)
![ViewCount](https://views.whatilearened.today/views/github/grahamwaters/what_would_doyle_do.svg)
![GitHub last commit](https://img.shields.io/github/last-commit/grahamwaters/what_would_doyle_do)
![GitHub repo size](https://img.shields.io/github/repo-size/grahamwaters/what_would_doyle_do)
![GitHub issues](https://img.shields.io/github/issues/grahamwaters/what_would_doyle_do)
![GitHub pull requests](https://img.shields.io/github/issues-pr/grahamwaters/what_would_doyle_do)

</div>


*By Graham Waters ~ A Machine Learning Experiment.*

## Introduction
The trick is how do I train a model on text data so that it learns about a person like Arthur Conan Doyle, for example? Further, how can I give meaningful insight into their life? That is the subject of this article and the primary purpose and problem statement of this repository and project.

---

**Problem Statement:** Can machine learning be applied to existing text data that an author writes or about a person that can help historical fiction authors write more accurately about their subject?

---

## Discussion:

The metrics for success are difficult to quantify in this project because the end result is an art that makes it somewhat nebulous. For this reason, I will winnow some of the more opaque tasks for components of this into bite-sized chunks.

First, consider what it means for a suggestion to be helpful in the context of historical fiction authorship. Let's say, for example, I am writing a book that is about Victorian England, and my characters all live in Edinburgh. I would like to know every detail there is to know about living in Edinburgh during the period that my novel spans. How does a machine learn about history? It learns by reading history books, but many have said, and this is a fair point, the victors always write that history, the people that successfully championed in their time, not by those who were unsuccessful or did not make a social impact. And so, any text data obtained from nonprimary sources (especially nonprimary sources that lack historical impact) is somewhat negligible. A stretch goal for this project is a more thorough analysis of primary sources. So let's look at another piece. Take, for example, a novel that was written in the years that we are considering. Specifically, consider The Strange Case of Dr. Jekyll and Mr. Hyde. This book is about a man that has two personalities, and he lives in Edinburgh. It was written by Robert Louis Stevenson. And it has a high vocabulary density, as do most books authored in that time period. What is the goal? The goal is to be able to ask your AI questions about an author or a historical figure and have the trained model(s) respond with some sort of useful feedback. One example is, "What do you think Abraham Lincoln would have said about poverty in England if I asked him?". I could ask him (or his ML model) the question in this case. The machine learning algorithm should be able to look at all text data that it has on record for Abraham Lincoln and quickly decide which pieces of that text are relevant to the question. Extract these pieces and formulate an answer. The answer may not be perfectly accurate; we will never really know unless it was precisely quoted from the primary source. These are just guesses. But it would be interesting to ask the questions that pertain to our time to people that don't live in our time. In a way, we are creating an interactive past to do this; it would require a lot of interpolation as well because we need exact data on what subjective opinions these people may have had or thought. However, it is not beyond the realm of possibility. To use things like the fact that Arthur Conan Doyle's father went insane and was interned in an insane asylum as a valid point of view or something that informs his perspective when he writes about someone "insane," especially when we consider the date when the book was published. What happened in the life of the person who wrote the book before this book was written? And how did it change them? If you look at books that were written after that date? These are the questions that I would like to answer with this research study. Now up to now, this has been very broad, and it lacks the specificity of a good data science problem. To make it more specific and valuable, as well as even possible. I am narrowing down the subject matter. I want to do this in a way that makes it easier for the analysis to occur and simultaneously makes it accessible and usable by other authors. For my study, I will look at Arthur Conan Doyle. There are many books that were written by Arthur Conan Doyle, an autobiography, and many first-hand accounts written by Arthur Conan Doyle that are maintained online due to his high notoriety and respected class. I would like to be able to train my model to talk like him, not just in the way he wrote but also with all of his flaws and authentic personality. Something that we can't quite connect to in our day, and we can't ignore it either. I've narrowed the scope enough to get started on a data science project that learns from the past to inform writers in the present and enables them to depict history's most interesting figures accurately.

## Stage One: Define the Problem

Stage one: the data science process always begins with defining the problem. So what is the problem? The problem we look to solve in this case is whether we can build a model as close to Arthur Conan Doyle as possible. Can I train a model that has the personal preferences of Doyle? Did he hate to eat shellfish? Was he partial to red wine or white wine? These are complex challenges. But with the tools provided in spacey and NLTK, we can solve a couple of these problems using named entity recognition and spaces large model, which is admittedly large and expensive computationally. We can examine our data for entities. So we'll do that in a later step. But first, now that we have somewhat defined a nebulous problem, let's explore it. Because the next step in the data science process is gathering data, we will need to decide how to do this.

## Stage Two: Gather Data
What data can we collect about Arthur Conan Doyle that will enable us to solve this problem? We need every book he's ever written, which is around 80 books and is provided through the Gutenberg repository. These books are included in the Data folder as text files; second, I would like to have anything he wrote that was a first-hand account because this is where we will get his personal preferences and his turns of phrase, and maybe even his personal biases, which are probably the most important things to gather once we gather his diaries, journals. Things other people said about him are the next step. Now we want to gather any second-hand accounts of Doyle. Many people have done years of research on historical figures and repeating it seems like a useless task and honestly is a waste of precious resources. So in this step, we want to gather any biographies that were written about Arthur Conan Doyle and any articles that were written about him, especially if they were written about him in the time that he lived. And this might be most useful if we were to gather the names of all of his second-degree connections. If we think about it, in terms of a LinkedIn network, though, Doyle's second-degree connections are the ones that are most likely to have the most accurate depictions of his preferences. This is, of course, an assumption that I am making. Once we gather the names of his second-degree connections, I think it would be an excellent step to assign weight to their accounts if they authored anything.

### What Values do we give the data?
Any first-hand accounts (written by Arthur Conan Doyle) are 100% because they represent ground truth. Things he did say or believe.
Family accounts written by his relatives should likely get a rating of 90% for data credibility.
Sources written or spoken (in the case of speeches) by his closest friends are worth 75% in data credibility.
And then the rest of his second-degree connections are around 60%. All other accounts will give 30% credibility.

### The Doyle DataFrame (DDF)

The Doyle DataFrame (DDF) is a data frame that contains all of the data that we have gathered about Arthur Conan Doyle.

```python
import pandas as pd
data = []
for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid))
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    # get total vocabulary used in this book
    num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
    data.append([
        fileid.split('.')[0] # remove .txt from file name
        ,round(num_chars/num_words)
        ,round(num_words/num_sents)
        # total vocabulary used divide total words used
        ,round(num_vocab/num_words,2)
    ])
pattern_metrics = pd.DataFrame(data,columns=['author-book','chars_per_word','words_per_sentence','vocabulary_rate'])
pattern_metrics
```
The snippet above comes [from this source](https://towardsdatascience.com/book-writing-pattern-analysis-625f7c47c9ad) and shows one method of analysis directly applied to gutenberg books.


## Stage Three: Explore Data

Once all the books have been saved into pickle files we can access them much quicker and easier. We can also use the pickle files to create a dataframe that contains all of the books in one place. This will make it easier to explore the data. We may want to keep them decentralized however, as this will allow for scalable comparisons between books.

```python
import pickle
import pandas as pd
import os
import glob

book_list = []
for book in glob.glob("Data/*.pickle"):
    book_list.append(book)

df = pd.DataFrame(book_list,columns=['book'])
df.head()
```
The snippet above creates a dataframe from the pickle files. This dataframe can be used to explore the data. We can see the file name, author, and book title. We can also see the number of words in the book, the number of sentences, and the number of unique words in the book.

## Stage Four: Prepare Data for modeling with CVEC




## Stage Four: Model Data

We have gathered the data and we have explored the data. The next step is to model the data. This is where we will begin to build our model. We will start by building a model that will predict the author of a book. This is a very simple model, but it will help us to understand the data and how to approach our problem.
