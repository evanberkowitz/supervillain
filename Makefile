# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
# sphinx-toolbox conflicts with sphinx-tabs and sphinx-autodoc-typehints in the
# locked project environment, so we use --with to inject it at build time only.
SPHINXBUILD   ?= uv run --with sphinx-toolbox sphinx-build
SPHINXAUTOBUILD ?= uv run --with sphinx-toolbox --with sphinx-autobuild sphinx-autobuild
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Watch for changes and rebuild; serves on http://127.0.0.1:8000
livehtml:
	@$(SPHINXAUTOBUILD) "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

.PHONY: help livehtml Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
