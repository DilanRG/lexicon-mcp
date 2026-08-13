"""Protocol-independent runtime for the offline lexicon."""

from .locator import ActiveDataset, DatasetLocator
from .service import LexiconService

__all__ = ["ActiveDataset", "DatasetLocator", "LexiconService"]
