.. _quickstart:

Quickstart
==========

This section is for the impatient. For more comprehensive, step-by-step guides, please consult the :doc:`./sphinx_gallery_output/index`.

.. image:: ../../images/quickstart.png

.. code-block:: python

    import matplotlib.pyplot as plt

    from matplotlib_set_diagrams import EulerDiagram, VennDiagram

    fig, axes = plt.subplots(2, 4, figsize=(15, 5))

    for ii, SetDiagram in enumerate([EulerDiagram, VennDiagram]):

        # Initialise from a list of sets:
        SetDiagram.from_sets(
            [
                {"a", "b", "c", "d", "e"},
                {"e", "f", "g"},
            ],
            ax=axes[ii, 0])

        # Alternatively, initialise directly from pre-computed subset sizes.
        SetDiagram(
            {
                (1, 0) : 4, # {"a", "b", "c", "d"}
                (0, 1) : 2, # {"f", "g"}
                (1, 1) : 1, # {"e"}
            },
            ax=axes[ii, 1])

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

        SetDiagram.as_wordcloud(sets, ax=axes[ii, 2])

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
        ax=axes[ii, 3])

        # set row titles
        axes[ii, 0].annotate(
            SetDiagram.__name__,
            xy         = (0, 0.5),
            xycoords   = 'axes fraction',
            xytext     = (-10, 0),
            textcoords = "offset points",
            ha         = 'right',
            va         = 'center',
            fontsize   = 'large',
            fontweight = 'bold',
        )

    fig.tight_layout()
    plt.show()
