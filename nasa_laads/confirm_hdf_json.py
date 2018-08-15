import json
import pandas as pd
from argparse import ArgumentParser

p = ArgumentParser(
    description=("Given a list of jsons, assert that they all contain the same fields")
)
p.add_argument("jsons", nargs="+")
FLAGS = p.parse_args()

read = {}
for j in FLAGS.jsons:
    with open(j, "r") as f:
        j = json.load(j)
        for k in j:
            read[k] = j

df = pd.DataFrame.from_dict(read)
assert not df.isnull().any().any(), "Looks like there are missing fields"
