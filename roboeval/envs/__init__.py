"""Environment wrappers conforming to the Gymnasium API.

Wraps simulation environments (currently ``gym-aloha`` Transfer Cube; see PRD
Section 6.2) into a uniform interface for the evaluation engine. The
robustness perturbation suite (PRD Section 6.4) is also implemented as
environment wrappers in this subpackage.
"""

from __future__ import annotations
