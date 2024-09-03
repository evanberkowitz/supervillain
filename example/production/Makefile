
.PHONY: demo
demo: demo-scaling.pdf .WAIT demo-transition.pdf
# Takes about 11 minutes on my machine.

demo-scaling.pdf: production.py scaling.py demo-scaling.py
	python production.py demo-scaling.py
	python history.py demo-scaling.py --pdf demo-scaling-history.pdf
	python demo-scaling.py --pdf demo-scaling.pdf

demo-transition.pdf: production.py transition.py demo-transition.py
	python production.py demo-transition.py
	python history.py demo-transition.py --pdf demo-transition-history.pdf
	python demo-transition.py --pdf demo-transition.pdf

demo-tidy:
	$(RM) demo-{scaling,transition}*h5

demo-clean: demo-tidy
	$(RM) demo-{scaling,transition}*pdf

