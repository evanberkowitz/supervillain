
.PHONY: demo
demo: demo-scaling.pdf .WAIT demo-transition.pdf
# Takes about 11 minutes on my machine.

demo-scaling.pdf: production.py scaling.py demo-scaling.py
	python production.py demo-scaling.py
	python demo-scaling.py --pdf demo-scaling.pdf

demo-transition.pdf: production.py transition.py demo-transition.py
	python production.py demo-transition.py
	python demo-transition.py --pdf demo-transition.pdf


