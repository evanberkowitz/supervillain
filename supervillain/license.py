
from pathlib import Path

this = Path(__file__)

def copyright():
    with open(f"{this.parent.parent}/LICENSE","r") as l:
        return l.read()

def license():
    with open(f"{this.parent.parent}/LICENSES/GPL","r") as g:
        return g.read()
