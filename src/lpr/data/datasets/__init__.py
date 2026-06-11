"""Dataset registry: every source the pipeline knows how to download and convert.

Add a dataset = subclass LprDataset (implement download + iter_samples) and list it
here. keremberke is deliberately absent: it is a strict subset of rxg4e.
"""

from .base import LprDataset
from .ccpd import CCPD
from .crpd import CRPD
from .ir_lpr import IRLPR
from .kaggle_andrewmvd import KaggleAndrewMVD
from .open_images import OpenImagesVRP
from .openalpr import OpenALPR
from .roboflow_sets import RoboflowLHQOW, RoboflowRXG4E
from .uc3m_lp import UC3MLP

DATASETS: dict[str, type[LprDataset]] = {
    cls.key: cls
    for cls in (CCPD, CRPD, RoboflowRXG4E, RoboflowLHQOW, OpenImagesVRP, IRLPR, UC3MLP, KaggleAndrewMVD, OpenALPR)
}
