import numpy as np
from utils.tiler import is_tissue

def test_is_tissue():
    white = np.full((224,224,3), 255, dtype=np.uint8)
    assert is_tissue(white, threshold=0.1) == False
