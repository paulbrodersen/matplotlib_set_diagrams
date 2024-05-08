#!/usr/bin/env python
"""
Word clouds
===========

To visualise subset contents (rather than sizes), use the
:code:`as_wordcloud` constructors, which generate a word cloud using
the word_cloud_ library.

_word_cloud: https://github.com/amueller/word_cloud)

"""

import matplotlib.pyplot as plt

from matplotlib_set_diagrams import (
    EulerDiagram,
    VennDiagram,
)

text_1 = """Lorem ipsum dolor sit amet, consectetur adipiscing elit,
sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut
enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi
ut aliquip ex ea commodo consequat."""

text_2 = """Duis aute irure dolor in reprehenderit in voluptate velit
esse cillum dolore eu fugiat nulla pariatur. Lorem ipsum dolor sit
amet."""


def word_tokenize(text):
    """Break a string into its constituent words, and convert the words
    into their 'standard' form (tokens).

    The procedure below is a poor-man's tokenization.
    Consider using the Natural Language Toolkit (NLTK) instead:

    >>> import nltk; words = nltk.word_tokenize(text)

    """

    # get a word list
    words = text.split(' ')
    # remove non alphanumeric characters
    words = [''.join(ch for ch in word if ch.isalnum()) for word in words]
    # convert to all lower case
    words = [word.lower() for word in words]

    return words

# Tokenize strings.
sets = [set(word_tokenize(text)) for text in [text_1, text_2]]

fig, (ax1, ax2) = plt.subplots(1, 2)
EulerDiagram.as_wordcloud(sets, ax=ax1)
VennDiagram.as_wordcloud(sets, ax=ax2)
plt.show()
