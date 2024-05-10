.. _installation:

Installation & Testing
======================

Install the current release of :code:`matplotlib_set_diagrams` with:

.. code-block:: shell

    pip install matplotlib_set_diagrams

To upgrade to a newer version, use the :code:`--upgrade` flag:

.. code-block::

    pip install --upgrade matplotlib_set_diagrams

If you do not have permission to install software systemwide, you can install into your user directory using the --user flag:

.. code-block::

    pip install --user matplotlib_set_diagrams

Alternatively, you can manually download matplotlib_set_diagrams from GitHub_ or PyPI_.
To install one of these versions, unpack it and run the following from the top-level source directory using the terminal:

.. _GitHub: https://github.com/paulbrodersen/matplotlib_set_diagrams
.. _PyPi: https://pypi.org/project/matplotlib_set_diagrams/

.. code-block::

    pip install .

For automated testing, install the additional dependencies required for testing using:

.. code-block::

    pip install matplotlib_set_diagrams[tests]

The test suite extensively uses baseline images to assert correctness.
Unfortunately, these are dependent on the local Matplotlib configuration and the available fonts,
and hence have to be user-generated.
Make sure to generate the baseline images **before making changes to the source code**:

.. code-block::

    pytest --mpl-generate-path=tests/baseline

The full test suite can then be executed by running the following from the top-level source directory using the terminal:

.. code-block::

    pytest --mpl
