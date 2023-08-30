# supervillain

Supervillain is a python package for studying the Villain model.

# Installation + Development

Navigate to the cloned repo and try

```
pip install .  # for development use pip install -e . 
./examples/villain-metropolis.py
```

If pip installation succeeds so too should the example script, which performs a simple Metropolis algorithm on the (φ, n) formulation of the model.

supervillain has documentation built with [sphinx](https://www.sphinx-doc.org/en/master/).
To build the documentation once you 

```
sphinx-build . _build
```

and then can open `_build/index.html` in a browser.
If you are developing you can replace `sphinx-build` with [`sphinx-autobuild`](https://pypi.org/project/sphinx-autobuild/) to get live updates to the documentation as you go.
