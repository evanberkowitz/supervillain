
.PHONY: demo
demo: demo-scaling.pdf .WAIT demo-transition.pdf
# Takes about 11 minutes on my machine.

.PHONY: demo-parallel
demo-parallel:
# Takes about 3-4 minutes on my machine.
	$(MAKE) PARALLEL=--parallel demo
# Join HDF5 files
	h5repack --merge demo/scaling-bootstrap.h5 demo/scaling-bootstrap.repacked.h5
	h5repack --merge demo/transition-bootstrap.h5 demo/transition-bootstrap.repacked.h5

demo-scaling.pdf: production.py scaling.py demo-scaling.py
	python production.py $(PARALLEL) demo-scaling.py
	python history.py demo-scaling.py --pdf demo-scaling-history.pdf
	python correlators.py demo-scaling.py --pdf demo-scaling-correlators.pdf
	python demo-scaling.py --pdf demo-scaling.pdf

demo-transition.pdf: production.py transition.py demo-transition.py
	python production.py $(PARALLEL) demo-transition.py
	python history.py demo-transition.py --pdf demo-transition-history.pdf
	python demo-transition.py --pdf demo-transition.pdf

.PHONY: demo-tidy
demo-tidy:
	$(RM) demo-{scaling,transition}*pdf

.PHONY: demo-clean
demo-clean: demo-tidy
	$(RM) demo/{scaling,transition}*h5

.PHONY: paper
paper: Z3-breaking.pdf

Z3-breaking.pdf: breaking.py Z3-breaking.py
	python production.py $(PARALLEL) Z3-breaking.py
	python breaking.py $(PARALLEL) Z3-breaking.py --pdf Z3-breaking.pdf

paper-tidy:
	$(RM) Z3-breaking.pdf

paper-clean: paper-tidy
	@ECHO "Clean the data files manually, it is too dangerous to provide a makefile target."
