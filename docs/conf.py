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
version = release = doc_version

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
    "tutorials-zero-to-thirty/README.md",
    "tutorials-weather-fed/README.md",
]

html_show_sourcelink = False
autodoc_typehints = "description"
autosummary_generate = True

# Options for HTML output
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = [
    "css/dfm-theme.css",
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
html_theme_options["footer_start"] = ["release-version", "copyright"]

# Multi-version docs: env-driven pydata version switcher. Activates only when
# HCT_DOCS_SWITCHER_URL is set (by the multi-version build job); a normal
# single-version build is unaffected. HCT_DOC_VERSION selects the active version.
#
# This block is part of a CROSS-PIPELINE contract shared with the HCT-managed
# `main` branch and MUST stay byte-for-byte equivalent to main's conf.py so all
# pages behave identically. See devdocs/MULTIVERSION-DOCS-RELEASE-BRANCH.md.
_switcher_url = os.environ.get("HCT_DOCS_SWITCHER_URL", "").strip()
# Only activate the switcher for a host-qualified URL ("//host/..." or an absolute
# "https://..."), mirroring HCT's own switcher_url validation. A host-less value like
# "/_static/switcher.json" (what a non-Pages/local build would produce if the URL were
# derived unconditionally) is treated as "no switcher" so local docs never render a
# dropdown that 404s. This guard lives here because conf.py is NOT HCT-generated, so it
# holds regardless of how the build script derives the URL.
if _switcher_url.startswith("//") or _switcher_url.startswith("https://"):
    # Fail fast on CI misconfiguration rather than publishing under the wrong
    # active version.
    _doc_version = os.environ.get("HCT_DOC_VERSION")
    if not _doc_version:
        raise RuntimeError(
            "HCT_DOC_VERSION must be set when HCT_DOCS_SWITCHER_URL is set "
            "(multi-version docs build)."
        )
    html_theme_options["switcher"] = {
        "json_url": _switcher_url,
        "version_match": _doc_version,
    }
    # REQUIRED: otherwise pydata fetches + json.loads() the switcher URL at build
    # time and fails when it doesn't exist yet (it 404s to an HTML page). The
    # dropdown is populated client-side at page load instead.
    html_theme_options["check_switcher"] = False
    html_theme_options.setdefault(
        "navbar_end", ["version-switcher", "navbar-icon-links"]
    )
    # Do NOT enable show_version_warning_banner: pydata only suppresses it for
    # the preferred version when versions are semver-comparable; "dev"/"latest"
    # are not, so it would show "unknown version" on every page.

favicons = ["favicon.ico"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]
