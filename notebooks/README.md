# Notebooks

`legacy/` contains the original notebooks as uploaded.

Recommended workflow:
1. Install the package (`pip install -e .`)
2. Create a new notebook that imports from `spnsa_feed.*` modules.

If you want, you can gradually refactor the legacy notebooks to use:
```python
from spnsa_feed.spnsa import spnsa
```
and the helpers in `spnsa_feed.data`, `spnsa_feed.feeds`, `spnsa_feed.experiments`, etc.
