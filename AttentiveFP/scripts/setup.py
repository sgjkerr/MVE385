#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os


# mkdir dirs
os.system("mkdir -p ../models")
os.system("mkdir -p ../results")

# check sklearn
try:
    import sklearn
except ImportError:
    raise ImportError("please install sklearn")

# check tensorflow
try:
    import tensorflow
except ImportError:
    raise ImportError("please install tensorflow")

# check deepchem
try:
    import deepchem
except ImportError:
    raise ImportError("please install deepchem")

# check dgl
try:
    import dgl
except ImportError:
    raise ImportError("please install dgl")
