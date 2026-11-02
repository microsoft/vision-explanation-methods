# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

"""Compatibility axes for Captum visualizations."""

from matplotlib.axes import Axes


class _CaptumAxes(Axes):
    """Support the legacy grid keyword used by Captum 0.5."""

    def grid(self, *args, **kwargs):
        """Forward legacy grid visibility as a positional argument."""
        if 'b' in kwargs:
            return super().grid(kwargs.pop('b'), *args, **kwargs)
        return super().grid(*args, **kwargs)
