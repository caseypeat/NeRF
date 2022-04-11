import sys
import yaml
from box import Box

if len(sys.argv) == 1:
    with open('./configs/config.yaml', 'r') as f:
        cfg = Box(yaml.safe_load(f))
elif len(sys.argv) == 2:
    with open(sys.argv[1], 'r') as f:
        cfg = Box(yaml.safe_load(f))