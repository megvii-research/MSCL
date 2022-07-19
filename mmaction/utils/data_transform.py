import imghdr
import msgpack
import lz4.frame
import cv2
import numpy as np

def imdecode(buf, flags=cv2.IMREAD_UNCHANGED):
    """\
    Decode an ordinaray or np4 image content to numpy array
    """
    if imghdr.what('', h=buf) is not None:
        return cv2.imdecode(np.frombuffer(buf, np.uint8), flags)

    try:
        pb = lz4.frame.decompress(buf)
        p = msgpack.unpackb(pb)
    except Exception:
        return None

    return np.frombuffer(p['d'], dtype=p['t']).reshape(p['s'])