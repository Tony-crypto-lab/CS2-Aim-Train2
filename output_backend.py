from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class OutputBackend(ABC):
    @abstractmethod
    def initialize(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def show_idle(self, rendered_image: np.ndarray, status_text: str = "") -> None:
        raise NotImplementedError

    @abstractmethod
    def show_lineup(self, lineup_info: dict[str, Any], rendered_image: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def show_status(self, text: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def resend_current(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_runtime_status(self) -> dict[str, Any]:
        raise NotImplementedError
