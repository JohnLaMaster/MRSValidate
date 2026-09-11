"""Sphinx configuration for the mrsvalidate documentation site."""

from __future__ import annotations

import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------
# The package lives in src/ (src-layout); make it importable for autodoc
# without requiring the package to already be installed in the build
# environment (Read the Docs also installs it via .readthedocs.yaml, but
# this keeps `sphinx-build` working from a bare checkout too).
sys.path.insert(0, os.path.abspath("../../src"))

import mrsvalidate  # noqa: E402

# -- Project information ------------------------------------------------------

project = "mrsvalidate"
author = "Ing. John T. LaMaster"
copyright = f"{datetime.now().year}, {author}"
version = mrsvalidate.__version__
release = version

# -- General configuration -----------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_gallery.gen_gallery",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- autodoc / autosummary -----------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autosummary_generate = True
add_module_names = False

# -- napoleon (NumPy-style docstrings) -----------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
# Render numpydoc "Attributes" sections as an `:ivar:` field list inside
# the class docstring rather than separate `.. attribute::` directives.
# Our dataclasses are also auto-documented by autodoc's own member
# listing (autodoc_default_options["members"]=True), so without this the
# two mechanisms describe the same attribute twice ("duplicate object
# description" warnings).
napoleon_use_ivar = True

# -- intersphinx ----------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "statsmodels": ("https://www.statsmodels.org/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- sphinx-gallery ---------------------------------------------------------------

sphinx_gallery_conf = {
    "examples_dirs": "examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"/plot_",
    "download_all_examples": False,
    "remove_config_comments": True,
    "matplotlib_animations": True,
    "image_scrapers": ("matplotlib",),
    "default_thumb_file": None,
    "doc_module": ("mrsvalidate",),
    "backreferences_dir": "gen_modules/backreferences",
    "reference_url": {"mrsvalidate": None},
}

# -- HTML output (pydata-sphinx-theme) -------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = f"{project} {version}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "github_url": "https://github.com/JohnLaMaster/MRSValidate",
    "navigation_with_keys": True,
    "show_nav_level": 2,
    "navigation_depth": 4,
    "collapse_navigation": False,
    "show_toc_level": 2,
    "use_edit_page_button": False,
    "navbar_align": "left",
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "logo": {
        "text": "mrsvalidate",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/JohnLaMaster/MRSValidate",
            "icon": "fa-brands fa-github",
        },
    ],
    # Light/dark mode toggle is enabled by default in pydata-sphinx-theme;
    # this pins the three-way (light/dark/auto) switcher explicitly so it
    # is never accidentally disabled by a future options change.
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}

html_context = {
    "github_user": "JohnLaMaster",
    "github_repo": "MRSValidate",
    "github_version": "main",
    "doc_path": "docs/source",
}

html_sidebars = {
    "**": ["sidebar-nav-bs"],
}

# -- Math rendering -----------------------------------------------------------
# sphinx.ext.mathjax (enabled above) renders every `:math:` role and
# `.. math::` directive via MathJax; no additional configuration is
# required for standard LaTeX-in-docstrings usage.

mathjax3_config = {
    "tex": {"macros": {}},
}
