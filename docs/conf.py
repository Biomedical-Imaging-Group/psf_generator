# -- Project information -----------------------------------------------------

project = 'PSF Library'
author = 'Yan Liu'

# The full version, including alpha/beta/rc tags
# release = '1.0.0'


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = ['.md', '.rst']
exclude_patterns = []

myst_enable_extensions = [
    'dollarmath',
    'amsmath',
    'deflist',
    'fieldlist',
]

myst_heading_anchors = 5
myst_dmath_allow_labels = True
myst_dmath_double_inline = True

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.builders.html',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinxcontrib.bibtex',
    'myst_nb',
    'autoapi.extension',
]

# Make sure the target is unique
autosectionlabel_prefix_document = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The master toctree document.
master_doc = 'index'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

bibtex_bibfiles = ['bibliography.bib']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/main/', None)
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = []

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}
html_theme = "default"
try:
    import pydata_sphinx_theme
    html_theme = "pydata_sphinx_theme"
    del pydata_sphinx_theme
except ModuleNotFoundError:
    pass


# The name for this set of Sphinx documents.
# "<project> v<release> documentation" by default.
html_title = ""

# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#
html_logo = "res/logo.png"

# The name of an image file (relative to this directory) to use as a favicon of
# the docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = "res/favicon.ico"

# Additional templates that should be rendered to pages, maps page names to
# template names.
html_additional_pages = {}

html_show_sourcelink = False

autoapi_dirs = ['../src']
autoapi_add_toctree_entry = True
autoapi_root = 'autoapi'
