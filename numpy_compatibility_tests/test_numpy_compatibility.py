# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

"""Tests for supported NumPy and Matplotlib versions."""

import matplotlib
import numpy as np

matplotlib.use('Agg')
import vision_explanation_methods  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402


def test_matplotlib_supports_numpy_1_and_2():
    """Verify Matplotlib plotting works with supported NumPy versions."""
    assert int(np.__version__.split('.')[0]) in (1, 2)
    assert vision_explanation_methods.__version__ is not None

    figure, axes = plt.subplots()
    axes.plot(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    plt.close(figure)
