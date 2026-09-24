#####################################################################################
#  The MIT License (MIT)
#
#  Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.
#####################################################################################

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re

html_theme = "rocm_docs_theme"
html_theme_options = {"flavor": "rocm-docs-home"}

templates_path = ["."]  # Use the current folder for templates

setting_all_article_info = True
all_article_info_os = ["linux"]

with open("../CMakeLists.txt", encoding="utf-8") as f:
    match = re.search(r".*\brocm_setup_version\(VERSION\s+([0-9.]+)[^0-9.]+", f.read())
    if not match:
        raise ValueError("VERSION not found!")
    version_number = match[1]

# for PDF output on Read the Docs
project = "MIGraphX"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number

extensions = [
    "rocm_docs",
    "rocm_docs.doxygen",
    "sphinx_collapse",
    "sphinxcontrib.datatemplates",
    "sphinx_substitution_extensions",
]

external_toc_path = "./sphinx/_toc.yml"
doxygen_root = "doxygen"
doxysphinx_enabled = False
doxygen_project = {
    "name": "doxygen",
    "path": "doxygen/xml",
}

substitutions_default_enabled = True
substitutions_hyperlink_targets_enabled = True

# Theme-related configs
html_title = f"{project} {version_number}"
html_theme = "rocm_docs_theme"
html_theme_options = {
    "flavor": "ai-ecosystem",
    "link_main_doc": True,
    "repository_url": "https://github.com/ROCm/AMDMIGraphX",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
}

# Publish the llms.txt index at the docs site root and let
# rocm-docs-core generate llms-full.txt after each build (the llms.txt standard,
# https://llmstxt.org/).
rocm_docs_generate_llms = True

external_projects_current_project = "amdmigraphx"
