import matplotlib.pyplot as plt

from matplotlib_set_diagrams import EulerDiagram, VennDiagram

fig, axes = plt.subplots(4, 2, figsize=(5, 10))

for ii, SetDiagram in enumerate([EulerDiagram, VennDiagram]):

    axes[0, ii].set_title(SetDiagram.__name__)

    # Initialise from a list of sets:
    SetDiagram.from_sets(
        [
            {"a", "b", "c", "d", "e"},
            {"e", "f", "g"},
        ],
        ax=axes[0, ii])

    # Alternatively, initialise directly from pre-computed subset sizes.
    SetDiagram(
        {
            (1, 0) : 4, # {"a", "b", "c", "d"}
            (0, 1) : 2, # {"f", "g"}
            (1, 1) : 1, # {"e"}
        },
        ax=axes[1, ii])

    # Visualise subset items as word clouds:
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

    SetDiagram.as_wordcloud(sets, ax=axes[2, ii])

    # The implementation generalises to any number of sets.
    # However, exact solutions are only guaranteed for two given sets,
    # and the more sets are given, the less likely it becomes that
    # the optimisation procedure finds even an approximate solution.
    # Furthermore, above four or five sets, diagrams become unintelligible.
    # Here an example of a 4-way set diagram:
    SetDiagram(
        {
            (1, 0, 0, 0) : 4.0,
            (0, 1, 0, 0) : 3.0,
            (0, 0, 1, 0) : 2.0,
            (0, 0, 0, 1) : 1.0,
            (1, 1, 0, 0) : 0.9,
            (1, 0, 1, 0) : 0.8,
            (1, 0, 0, 1) : 0.7,
            (0, 1, 1, 0) : 0.6,
            (0, 1, 0, 1) : 0.5,
            (0, 0, 1, 1) : 0.4,
            (1, 1, 1, 0) : 0.3,
            (1, 1, 0, 1) : 0.25,
            (1, 0, 1, 1) : 0.2,
            (0, 1, 1, 1) : 0.15,
            (1, 1, 1, 1) : 0.1,
        },
    ax=axes[3, ii])

fig.tight_layout()
fig.savefig("images/quickstart.png", dpi=300)
plt.show()
