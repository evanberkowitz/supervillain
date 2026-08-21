# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import subprocess
import sys

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
        'sphinx_toolbox.collapse',
        'sphinx_toolbox.github',
        'sphinx_toolbox.source',
        'sphinx_toolbox.sidebar_links',
        'sphinx_favicon',
        'sphinxcontrib.bibtex',
        'sphinx_git',
        'matplotlib.sphinxext.plot_directive',
]

# https://sphinx-toolbox.readthedocs.io/en/stable/extensions/github.html
github_username='evanberkowitz'
github_repository='supervillain'
source_link_target = 'GitHub'


def _git_branch():
    rtd = os.environ.get('READTHEDOCS_VERSION')
    if rtd:
        return rtd
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return 'main'


def setup(app):
    # sphinx_toolbox.github hard-codes "master"; use the branch being built.
    app.connect('config-inited', _set_github_source_url, priority=851)

    # sphinx_toolbox.collapse writes a labelled .. collapse:: as
    # <details name="the-label"> and never emits an id, so a :ref: to one
    # resolves at build time and lands nowhere in the browser.  Nothing warns:
    # the cross-reference itself is perfectly well formed, and only the anchor
    # it points at is missing.  Register a visitor that emits the ids.
    from sphinx_toolbox.collapse import CollapseNode, depart_collapse_node
    app.add_node(
            CollapseNode,
            html=(_visit_collapse_node, depart_collapse_node),
            override=True,
            )


def _visit_collapse_node(translator, node):
    # sphinx_toolbox's own visitor, plus the ids; see setup().
    from html import escape
    from sphinx_toolbox.collapse import CollapseSummaryNode

    tag = ['details']
    if node.get('ids'):
        # An element carries one id.  A directive given several labels would
        # lose all but the first, which no page here does.
        tag.append(f'id="{node["ids"][0]}"')
    if node.get('names'):
        tag.append('name="{}"'.format(' '.join(node['names'])))
    if node.get('classes'):
        tag.append('class="{}"'.format(' '.join(node['classes'])))
    if node.attributes.get('open', False):
        tag.append('open')

    translator.body.append('<{}>\n'.format(' '.join(tag)))
    if not any(isinstance(child, CollapseSummaryNode) for child in node.children):
        translator.body.append(f"<summary>{escape(node.get('label') or '')}</summary>")
    translator.context.append('</details>')


def _set_github_source_url(app, config):
    config.github_source_url = config.github_url / 'blob' / _git_branch()

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

autodoc_default_options = {
    'member-order': 'bysource',
    'special-members': '__call__',
    'undoc-members': True,
}

todo_include_todos=True

napoleon_use_param=False #see https://github.com/sphinx-doc/sphinx/issues/10330
