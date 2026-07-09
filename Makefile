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
#
# Targets carrying a "## " comment are listed automatically, so a new target
# documents itself.  The catch-all below hands every *other* target to Sphinx as
# a builder name, which is why the builders get their own listing: ask Sphinx.
help: ## Show this message
	@echo 'Usage: make [target]'
	@echo
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## /{printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo 'Any other target is forwarded to Sphinx as a builder (html, latexpdf, ...).'
	@echo 'Run "make sphinx-help" to list those.'

# Sphinx's own help, i.e. the builders it can run.  This needs an explicit rule:
# the catch-all would forward the literal word "sphinx-help" to -M, and Sphinx
# would reject it as an unknown builder.  What Sphinx wants is -M help.
sphinx-help: ## List the builders Sphinx provides (html, latexpdf, linkcheck, ...)
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Watch for changes and rebuild; serves on http://127.0.0.1:8000
#
# SOURCEDIR is the project root, so the watcher sees the whole tree.  Two things
# must be excluded or the build retriggers itself endlessly:
#
#   --ignore "$(BUILDDIR)"      The build output.  Use the bare dir, NOT
#                              "$(BUILDDIR)/*": sphinx-autobuild fnmatches each
#                              --ignore against the absolute path, and the glob
#                              matches files *inside* the dir but not the dir
#                              entry itself (which macOS fsevents reports on
#                              every write).  "$(BUILDDIR)" matches both.
#
#   --re-ignore "__pycache__"  THE main culprit.  Building imports the package
#                              and runs the .. plot:: scripts, which compile the
#                              numba @njit(cache=True) kernels.  numba writes its
#                              cache (and Python its bytecode) into source-tree
#                              __pycache__ dirs via tmp<random> temp files --
#                              outside $(BUILDDIR), so each build's writes
#                              retrigger the next.  Ignore all __pycache__ paths.
#
# (conf.py's exclude_patterns only affects which sources are parsed, not what the
# watcher monitors.)
livehtml: ## Rebuild on change and serve on http://127.0.0.1:8000
	@$(SPHINXAUTOBUILD) "$(SOURCEDIR)" "$(BUILDDIR)/html" --ignore "$(BUILDDIR)" --re-ignore "__pycache__" $(SPHINXOPTS) $(O)

# Clear numba's on-disk kernel cache (*.nbc / *.nbi under the source tree).
#
# The @njit(cache=True) kernels in supervillain/lattice/_kernels.py persist
# compiled variants into source-tree __pycache__ dirs.  On this pre-release
# stack (py3.14 + numba 0.65 + numpy 2.x) numba's cache invalidation does not
# reliably fire when numpy is bumped, so a variant compiled against an old numpy
# can be served against a new one and blow up with
#   RuntimeError: In 'NRT_adapt_ndarray_to_python', 'descr' is NULL
# Run this after any dependency change that moves numpy or numba.
clean-numba: ## Delete numba's on-disk kernel cache (*.nbc / *.nbi)
	find . \( -name '*.nbc' -o -name '*.nbi' \) -delete

.PHONY: help sphinx-help livehtml clean-numba Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
