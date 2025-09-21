# Configuration file for the Sphinx documentation builder.
import os
import sys
# Add the repository root (parent directory of docs/) to sys.path so Sphinx can import the package.
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

project = 'STAM'
author = "Na'ama Hallakoun"
release = '0.1'
version = '0.2.2'

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'

# Generate autosummary stub pages automatically
autosummary_generate = True

# Autodoc defaults: include members and show inheritance; preserve member order
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'inherited-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',
}
autodoc_member_order = 'bysource'
add_module_names = False

# Napoleon (Google/Numpy style) settings to include full docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_param = True
napoleon_use_rtype = True

# Show type hints in the description rather than in the signature
autodoc_typehints = 'description'
