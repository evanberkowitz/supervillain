# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'supervillain'
copyright = '2023, Berkowitz, Buesing, Cherman, Jacobson, and Sen'
author = 'Berkowitz, Buesing, Cherman, Jacobson, and Sen'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
        'sphinx.ext.todo',
        'sphinx.ext.napoleon',
        'sphinx_math_dollar',
        'sphinx.ext.mathjax',
        'sphinx.ext.autodoc',
        'sphinx.ext.viewcode',
        'sphinx_toolbox.collapse',
        'sphinx_toolbox.github',
        'sphinx_toolbox.sidebar_links',
        'sphinxcontrib.bibtex',
        'sphinx_git',
        'matplotlib.sphinxext.plot_directive',
]

# https://sphinx-toolbox.readthedocs.io/en/stable/extensions/github.html
github_username='evanberkowitz'
github_repository='supervillain'

templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'setup.py']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

bibtex_bibfiles = ['master.bib']
bibtex_default_style = 'unsrt'
bibtex_reference_style = 'label'

autodoc_default_options = {
    'member-order': 'bysource',
    'special-members': '__call__',
    'undoc-members': True,
}

todo_include_todos=True
napoleon_use_param=False #see https://github.com/sphinx-doc/sphinx/issues/10330
