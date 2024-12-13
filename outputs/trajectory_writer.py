"""Module used to write output to file."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from typing_extensions import Self

if TYPE_CHECKING:
    import types

    import numpy as np


class TrajectoryWriter:
    """Class for displaying 2D image."""

    def __init__(self, filename: str) -> None:
        """Construct."""
        self.file = None
        self.filename = filename

    def __enter__(self) -> Self:
        """Enter the context manager."""
        self.file = Path(self.filename).open("w", encoding="utf-8")
        return self

    def write(self, timestamp: int, pose: np.ndarray) -> None:
        """Write the pose to the txt file."""
        quat = pose[3:]

        seconds = timestamp / 1000

        if self.file:
            # TUM format: timestamp x y z qx qy qz qw
            line = f"{seconds:.4f} "
            line += f"{pose[0]} {pose[1]} {pose[2]} "
            line += f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}\n"
            self.file.write(line)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        """Exit the context manager and close the file."""
        if self.file:
            self.file.close()
            self.file = None
