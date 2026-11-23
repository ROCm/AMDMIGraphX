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

templates_path = ["."]  # Use the current folder for templates

setting_all_article_info = True
all_article_info_os = ["linux"]

with open("../CMakeLists.txt", encoding="utf-8") as f:
    match = re.search(r".*\brocm_setup_version\(VERSION\s+([0-9.]+)[^0-9.]+",
                      f.read())
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
    "rocm_docs.selector",
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
# Substituting into hyperlink targets needs sphinx-substitution-extensions
# >=2026.8.2, which requires Python >=3.11 and Sphinx >=8.1; RTD builds this
# project on Python 3.10. Until that moves, spell the ROCm version out in
# external URLs instead of using |ROCM_VERSION|.
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

# Bump these when the docs target a new ROCm or Torch-MIGraphX release; every
# ROCm-versioned string used below (and in the install docs, via
# myst_substitutions) is derived from them so a release bump only touches
# this block. MIGRAPHX_VERSION below is parsed from ../CMakeLists.txt
# instead, since that file is the source of truth for the MIGraphX version.
rocm_version = "10.2.0"
rocm_version_pkg_suffix = "10.2"
rocm_major = "10"
rocm_docs_version = f"docs-{rocm_version}"
torch_migraphx_version = "1.2"

myst_substitutions = {
    "ROCM_VERSION":
    rocm_version,
    "ROCM_VERSION_PKG_SUFFIX":
    rocm_version_pkg_suffix,
    "ROCM_MAJOR":
    rocm_major,
    "MIGRAPHX_VERSION":
    version,
    "TORCH_MIGRAPHX_VERSION":
    torch_migraphx_version,
    "WHL":
    f'"torch_migraphx=={torch_migraphx_version}+rocm{rocm_version}"',
    "WHL_PKG_REPO":
    "https://stable.repo.amd.com/rocm/migraphx/whl-next/",
    "TARBALL_REPO":
    "https://stable.repo.amd.com/rocm/migraphx/tarball/",
    "TARBALL_NAME":
    f"amdrocm{rocm_major}-migraphx-{version}.tar.gz",
    "ROCM_INSTALL_DOC":
    f"`Install AMD ROCm {rocm_version} <https://rocm.docs.amd.com/en/{rocm_docs_version}/install/rocm.html>`__",
    "ROCM_COMPAT_DOC":
    f"`ROCm {rocm_version} compatibility matrix <https://rocm.docs.amd.com/en/{rocm_docs_version}/compatibility/compatibility-matrix.html>`__",
    "TORCH_INSTALL_DOC":
    f"`Install PyTorch for ROCm <https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html?rocm-ver={rocm_version}>`__",
}
rst_prolog = "\n".join(f".. |{key}| replace:: {val}"
                       for key, val in myst_substitutions.items())
