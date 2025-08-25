
import numpy as np

# Code adapted from: https://github.com/facebookresearch/Mask2Former/blob/main/demo/flow_vis.py
# Apache-2.0 License

TAG_CHAR = "PIEH"

def read_flow(file_path):
    """
    Read a .flo file.
    """
    with open(file_path, "rb") as f:
        tag = f.read(4).decode("utf-8")
        if tag != TAG_CHAR:
            raise ValueError(f"Wrong tag in .flo file: {tag}")
        
        width = np.fromfile(f, np.int32, 1)[0]
        height = np.fromfile(f, np.int32, 1)[0]
        
        data = np.fromfile(f, np.float32, count=height * width * 2)
        data.resize((height, width, 2))
        return data

def write_flow(file_path, flow):
    """
    Write a .flo file.
    """
    height, width, n_bands = flow.shape
    if n_bands != 2:
        raise ValueError(f"Flow must have 2 bands, has {n_bands}")

    with open(file_path, "wb") as f:
        f.write(TAG_CHAR.encode("utf-8"))
        np.array([width, height], dtype=np.int32).tofile(f)
        flow.astype(np.float32).tofile(f)

