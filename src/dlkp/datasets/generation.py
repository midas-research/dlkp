import os, sys
from dataclasses import dataclass, field
from typing import Optional
from datasets import ClassLabel, load_dataset
from . import KpDatasets


class KpGenerationDatasets(KpDatasets):
    def __init__(self) -> None:
        super().__init__()
        pass
