# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

# Load environment variables
doc_version = os.getenv("DOC_VERSION", "main")
logging.info(f"Documentation version: {doc_version}")

# Project information
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = "NVIDIA Data Federation Mesh"
copyright = "2026, NVIDIA"
author = "NVIDIA"

# General configuration
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_favicon",
    "myst_nb",
    "sphinx_design",
    "sphinx_togglebutton",
    "sphinx_copybutton",
]

source_suffix = [".rst", ".md", ".ipynb"]
myst_enable_extensions = ["colon_fence"]
myst_heading_anchors = 4
nb_execution_mode = "off"
nb_output_stderr = "remove"
templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "sphinxext.py",
    "Thumbs.db",
    ".DS_Store",
    "tutorials-example-fed",
]

html_show_sourcelink = False
autodoc_typehints = "description"
autosummary_generate = True

# Options for HTML output
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = [
    "css/nvidia-sphinx-theme.css",
]
html_theme_options = {
    "logo": {
        "text": "NVIDIA Data Federation Mesh",
        "image_light": "_static/NVIDIA-Logo-V-ForScreen-ForLightBG.png",
        "image_dark": "_static/NVIDIA-Logo-V-ForScreen-ForDarkBG.png",
    },
    "navbar_align": "content",
    "navbar_start": [
        "navbar-logo",
    ],
    # "switcher": {
    #     "json_url": "https://raw.githubusercontent.com/NVIDIA/data-federation-mesh/gh-pages/_static/switcher.json",
    #     "version_match": doc_version,  # Set DOC_VERSION env variable to change
    # },
    # "external_links": [
    #     {
    #         "name": "Recipes",
    #         "url": "https://github.com/NVIDIA/data-federation-mesh/tree/main/recipes",
    #     },
    #     {
    #         "name": "Changelog",
    #         "url": "https://github.com/NVIDIA/data-federation-mesh/blob/main/CHANGELOG.md",
    #     },
    # ],
    "icon_links": [
        {
            # Label for this link
            "name": "Github",
            # URL where the link will redirect
            "url": "https://github.com/NVIDIA/data-federation-mesh",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-square-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        }
    ],
}
favicons = ["favicon.ico"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]
