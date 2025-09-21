# This file proxies to the parent docs/conf.py so Sphinx can be run from the
# source directory which expects a conf.py to be present there.
import importlib.util
import os
import sys

# Load the parent docs/conf.py as a module named 'docs_conf' to avoid
# executing it in the wrong globals context which Sphinx checks for.
conf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'conf.py'))
spec = importlib.util.spec_from_file_location('docs_conf', conf_path)
docs_conf = importlib.util.module_from_spec(spec)
sys.modules['docs_conf'] = docs_conf
spec.loader.exec_module(docs_conf)

# Copy attributes from the loaded module into this module's globals so Sphinx
# finds the expected configuration variables.
globals().update({k: getattr(docs_conf, k) for k in dir(docs_conf) if k.isupper() or k in (
	'project', 'author', 'version', 'release', 'extensions', 'templates_path', 'exclude_patterns', 'html_theme', 'intersphinx_mapping', 'autosummary_generate'
)})
