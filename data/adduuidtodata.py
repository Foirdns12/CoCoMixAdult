import os

import numpy as np
import pandas as pd
import random
import uuid
import hashlib

PATH = os.path.dirname(os.path.abspath(__file__))
#rd = random.Random()
#rd.seed(42)
#uuid.uuid4 = lambda: uuid.UUID(int=rd.getrandbits(128))

df = pd.ExcelFile(os.path.join(PATH, "immodata", "immo1.5fix.xlsx")).parse(0)
df['factID'] = df.apply(lambda x: hashlib.md5("".join(map(str, x)).encode()).hexdigest(),
                                                    axis=1)
#df['factID']=uuid.uuid4()
df.to_csv(os.path.join(PATH, "immodata", "immo1.6fix.csv"))