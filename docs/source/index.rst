.. Matplotlib set diagrams documentation master file, created by
   sphinx-quickstart on Thu May  9 15:55:35 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Matplotlib Set Diagrams
=======================

*Draw Euler diagrams and Venn diagrams with Matplotlib.*

Euler_ and Venn_ diagrams are used to visualise the relationships between sets. Both typically employ circles to represent sets, and areas where two circles overlap represent subsets common to both supersets.
Venn diagrams show all possible relationships of inclusion and exclusion between two or more sets.
In Euler diagrams, the area corresponding to each subset is scaled according to the size of the subset. If a subset doesn't exist, the corresponding area doesn't exist.

.. _Euler: https://en.wikipedia.org/wiki/Euler_diagram
.. _Venn: https://en.wikipedia.org/wiki/Venn_diagram


Contributing & Support
----------------------

If you get stuck and have a question that is not covered in the documentation, please raise an issue on GitHub_.
If applicable, make a sketch of the desired result.
If you submit a bug report, please make sure to include the complete error trace.
Include any relevant code and data in a `minimal, reproducible example`__.
Pull requests are always welcome.

.. _GitHub: https://github.com/paulbrodersen/matplotlib_set_diagrams/issues
__ https://stackoverflow.com/help/minimal-reproducible-example


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Documentation

   installation.rst
   quickstart.rst
   sphinx_gallery_output/index.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API Reference

   eulerdiagram.rst
   venndiagram.rst
   utils.rst
