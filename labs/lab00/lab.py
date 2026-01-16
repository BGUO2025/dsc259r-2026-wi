# lab.py


from pathlib import Path
import io
import pandas as pd
import numpy as np
np.set_printoptions(legacy='1.21')


# ---------------------------------------------------------------------
# QUESTION 0
# ---------------------------------------------------------------------


def env_check():
    import pandas as pd
    import otter 
    import sklearn 
    print(f'pandas: {pd.__version__}')
    print(f'otter: {otter.__version__}')
    print(f'scikit-learn: {sklearn.__version__}')
