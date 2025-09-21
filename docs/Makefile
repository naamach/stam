# Minimal Sphinx Makefile

SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = _build

.PHONY: help html clean

help:
	@echo "Makefile for Sphinx docs"

html:
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	rm -rf $(BUILDDIR)
