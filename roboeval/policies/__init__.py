"""Policy loaders and inference wrappers.

Loads pretrained policies from the LeRobot model zoo and exposes a uniform
``select_action`` interface for the evaluation engine. v1.0 supports the ACT
checkpoint ``lerobot/act_aloha_sim_transfer_cube_human`` (PRD Section 6.1).
"""

from __future__ import annotations
