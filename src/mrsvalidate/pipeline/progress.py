"""Minimal progress reporting for batch runs (terminal, IDE, notebook, CI)."""

from __future__ import annotations

from typing import Protocol

__all__ = ["NullProgressReporter", "PrintProgressReporter", "ProgressReporter"]


class ProgressReporter(Protocol):
    """Structural interface for reporting batch-run progress.

    Any object exposing ``on_step_start``/``on_step_end`` with this
    signature satisfies the protocol — implementations need not
    inherit from it explicitly.

    Examples
    --------
    >>> from mrsvalidate.pipeline import ProgressReporter, PrintProgressReporter
    >>> reporter: ProgressReporter = PrintProgressReporter()
    >>> reporter.on_step_start("regression", 1, 2)
    [1/2] running regression...
    """

    def on_step_start(self, name: str, index: int, total: int) -> None:
        """Called immediately before a step begins running.

        Parameters
        ----------
        name : str
            Name of the step about to run.
        index : int
            One-based position of this step within the run.
        total : int
            Total number of steps in the run.

        Returns
        -------
        None
        """
        ...

    def on_step_end(self, name: str, status: str) -> None:
        """Called immediately after a step finishes running.

        Parameters
        ----------
        name : str
            Name of the step that just finished.
        status : str
            Outcome status string for the step (e.g. ``"valid"`` or
            ``"failed_exception"``).

        Returns
        -------
        None
        """
        ...


class NullProgressReporter:
    """Progress reporter that reports nothing.

    The default for library use inside another program, where the
    caller does not want batch-run progress printed on its behalf.

    Examples
    --------
    >>> from mrsvalidate.pipeline import NullProgressReporter
    >>> reporter = NullProgressReporter()
    >>> reporter.on_step_start("regression", 1, 1)
    >>> reporter.on_step_end("regression", "valid")
    """

    def on_step_start(self, name: str, index: int, total: int) -> None:
        pass

    def on_step_end(self, name: str, status: str) -> None:
        pass


class PrintProgressReporter:
    """Simple stdout progress reporter for terminal/notebook/CI use.

    Examples
    --------
    >>> from mrsvalidate.pipeline import PrintProgressReporter
    >>> reporter = PrintProgressReporter()
    >>> reporter.on_step_start("regression", 1, 1)
    [1/1] running regression...
    >>> reporter.on_step_end("regression", "valid")
        regression: valid
    """

    def on_step_start(self, name: str, index: int, total: int) -> None:
        print(f"[{index}/{total}] running {name}...")

    def on_step_end(self, name: str, status: str) -> None:
        print(f"    {name}: {status}")
