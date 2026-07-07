# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

from docutils import nodes
from docutils.parsers.rst import roles

sys.path.insert(0, os.path.abspath('.'))

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
        'sphinx_favicon',
        'sphinxcontrib.bibtex',
        'sphinx_git',
        'matplotlib.sphinxext.plot_directive',
]

templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'setup.py', '.venv']


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

# Standalone illustrated reports.  These are hand-authored HTML artifacts rather
# than Sphinx source files, so copy them beside the built documentation and link
# to them from the No-Intersection model page.
html_extra_path = [
    'worm-algorithms.html',
    'spun-trefoil.html',
    'fable-worm-replacement-report.html',
]

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'css/math.css',
]

favicons = [
        'favicon/favicon.ico',
        'favicon/favicon-16x16.png',
        'favicon/favicon-32x32.png',
        {
            'rel':  'apple-touch-icon',
            'href': 'favicon/apple-touch-icon.png',
        },
        'favicon/android-chrome-192x192.png',
        'favicon/android-chrome-512x512.png',
    ]

bibtex_bibfiles = ['master.bib']
bibtex_default_style = 'unsrt'
bibtex_reference_style = 'label'


def source_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    """Render legacy ``:source:`` references without sphinx-toolbox."""
    options = dict(options or {})
    options.setdefault('classes', []).append('literal')
    return [nodes.literal(rawtext, text, **options)], []


roles.register_local_role('source', source_role)

autodoc_default_options = {
    'member-order': 'bysource',
    'special-members': '__call__',
    'undoc-members': True,
}

todo_include_todos=True

napoleon_use_param=False #see https://github.com/sphinx-doc/sphinx/issues/10330
