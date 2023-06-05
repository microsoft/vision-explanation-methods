# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines constants for vision-explanation-methods."""

from enum import Enum


class Device(Enum):
    """Specifies all possible device types."""

    CPU = 'cpu'
    CUDA = 'cuda'
    AUTO = 'auto'
