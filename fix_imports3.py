import re
with open("tests/test_viz.py", "r") as f:
    content = f.read()

content = content.replace("from unittest.mock import MagicMock, patch\n\nfrom torchregress.viz.utils import create_grid_figure", "")
content = "from unittest.mock import MagicMock, patch\nfrom torchregress.viz.utils import create_grid_figure\n" + content

with open("tests/test_viz.py", "w") as f:
    f.write(content)
