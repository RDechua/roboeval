# Policies

The `Policy` protocol (declares `policy_id` + `device` + `select_action`) plus a thin LeRobot ACT
adapter. The policy factory swaps implementations behind a single `kind:` config flag.

::: roboeval.policies
