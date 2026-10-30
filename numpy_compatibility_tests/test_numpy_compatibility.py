# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

"""Tests for supported NumPy and Matplotlib versions."""

from types import SimpleNamespace

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import vision_explanation_methods  # noqa: E402
import vision_explanation_methods.DRISE_runner as dr  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402
from vision_explanation_methods.evaluation.pointing_game import \
    PointingGame  # noqa: E402


def test_matplotlib_supports_numpy_1_and_2():
    """Verify Matplotlib plotting works with supported NumPy versions."""
    assert int(np.__version__.split('.')[0]) in (1, 2)
    assert vision_explanation_methods.__version__ is not None

    figure, axes = plt.subplots()
    axes.plot(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    plt.close(figure)


def test_pointing_game_visualization_supports_numpy_1_and_2(tmp_path):
    """Render a VEM saliency visualization with modern Matplotlib."""
    image = Image.new('RGB', (4, 4))
    saliency_scores = torch.ones(1, 4, 4)
    figure = PointingGame(None, device='cpu').visualize_highly_salient_pixels(
        image, saliency_scores, gt_bbox=[0, 0, 2, 2])
    try:
        output = tmp_path / 'pointing_game.png'
        figure.savefig(output)
        assert output.stat().st_size > 0
    finally:
        plt.close(figure)


def test_drise_visualization_supports_numpy_1_and_2(tmp_path, monkeypatch):
    """Render D-RISE output without downloading or running a model."""
    image_path = tmp_path / 'image.png'
    Image.new('RGB', (4, 4)).save(image_path)
    detections = [SimpleNamespace(class_scores=torch.tensor([[0.0, 1.0]]))]
    model = SimpleNamespace(predict=lambda image: detections)
    monkeypatch.setattr(
        dr.drise, 'DRISE_saliency',
        lambda **kwargs: [[{'detection': torch.ones(1, 4, 4)}]])
    save_name = str(tmp_path / 'drise')
    try:
        figures, output, labels = dr.get_drise_saliency_map(
            str(image_path), save_name, model=model, num_masks=1,
            device_choice='cpu', max_figures=1)
        assert len(figures) == 1
        assert output == save_name
        assert labels == [1]
        assert (tmp_path / 'drise0.jpg').stat().st_size > 0
    finally:
        plt.close('all')
