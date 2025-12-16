# Setup

This project depends on data provided by the researchers of the original study.
It is directly sourced from their Dropbox and can be loaded using the `setup.py`
script.

The script can be run in the root of this project like this:
```sh
python setup/setup.py
```

Additionally, the function is included in all Jupyter notebooks to ensure the files
are loaded before running the experiments.

```python
from setup.setup import load_data
load_data()
```