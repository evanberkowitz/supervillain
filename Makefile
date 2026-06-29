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
livehtml:
	@$(SPHINXAUTOBUILD) "$(SOURCEDIR)" "$(BUILDDIR)/html" --ignore "$(BUILDDIR)" --re-ignore "__pycache__" $(SPHINXOPTS) $(O)

.PHONY: help livehtml Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
