from typing import Optional

import attr
import numpy as np

# This class allows to evaluate the SI-SNR performance measures for one or a list of WAV files


@attr.s(auto_attribs=True)
class Evaluator:
    results: np.ndarray
    mean_some_metric: Optional[float] = None

    def evaluate_single_sample(self):
        """Return value of some loss fnc"""
        pass

    def evaluate_list(self):
        """Return value of some loss fnc"""
        pass

    def save_results(self, file_path):
        np.savetxt(file_path, X=self.results, fmt="%2.1f")
