# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Matplotlib Set Diagrams'
copyright = '2024, Paul Brodersen'
author = 'Paul Brodersen'

# The full version, including alpha/beta/rc tags
import matplotlib_set_diagrams
release = matplotlib_set_diagrams.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'numpydoc',
    'sphinx_gallery.gen_gallery',
]

autosummary_generate = True
numpydoc_show_class_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Options for Sphinx Gallery ------------------------------------------------

from sphinx_gallery.sorting import FileNameSortKey

sphinx_gallery_conf = {
    'examples_dirs': ['../../examples'],                                     # path to your example scripts
    'gallery_dirs': ['sphinx_gallery_output', 'sphinx_gallery_animations'],  # path to where to save gallery generated output
    'within_subsection_order': FileNameSortKey,
    'matplotlib_animations': True,
}

# -- Suppress warnings --------------------------------------------------------

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
