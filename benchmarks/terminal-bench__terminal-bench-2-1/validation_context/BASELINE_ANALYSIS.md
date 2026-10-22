# Qwen3.5-9B-Base baseline on the fixed TBLite validation set

This pack describes the initial model run only. It is diagnostic context, not training data.
Verifier tests, verifier output, solutions, patches, and reference answers are intentionally absent.

## Summary
- Fixed tasks: 50
- Accuracy: 0.120000 (6/50 solved)
- Mean recorded reward: 0.162690
- solved: 6
- partial: 3
- failed: 26
- infrastructure_error: 15
- missing_trajectory: 0
- ungraded: 0

## Initial failure signals
- 26 tasks reached a clean verifier verdict with zero reward; inspect these first for incorrect implementation or incomplete verification behavior.
- 15 tasks ended with a Harbor error; the recorded exception type and trajectory distinguish timeout/context growth from a clean task failure.
- 3 tasks earned partial reward and are the clearest near-term improvement candidates.
- 6 tasks were solved and provide positive examples of successful tool-loop behavior.

`infrastructure_error` and `missing_trajectory` are not treated as clean model failures.
Use each task's instruction and Agent trajectory to inspect tool use, debugging, looping,
context growth, and stopping behavior. Do not infer hidden verifier details from the reward.

## Per-task index

| # | Task | Status | Reward | Instruction | Trajectory |
|---:|---|---|---:|---|---|
| 1 | `application-debug` | failed | 0.000000 | [task](tasks/application-debug/instruction.md) | [trajectory](trajectories/application-debug.json) |
| 2 | `pgn-chess-repair-puzzles` | partial | 0.700000 | [task](tasks/pgn-chess-repair-puzzles/instruction.md) | [trajectory](trajectories/pgn-chess-repair-puzzles.json) |
| 3 | `neutron-submission` | infrastructure_error | 0.000000 | [task](tasks/neutron-submission/instruction.md) | [trajectory](trajectories/neutron-submission.json) |
| 4 | `cosign-keyless-signing` | infrastructure_error | 0.000000 | [task](tasks/cosign-keyless-signing/instruction.md) | [trajectory](trajectories/cosign-keyless-signing.json) |
| 5 | `vimscript-vim-quine` | failed | 0.000000 | [task](tasks/vimscript-vim-quine/instruction.md) | [trajectory](trajectories/vimscript-vim-quine.json) |
| 6 | `publisher-market-analysis` | failed | 0.000000 | [task](tasks/publisher-market-analysis/instruction.md) | [trajectory](trajectories/publisher-market-analysis.json) |
| 7 | `word-derangement-mapping` | failed | 0.000000 | [task](tasks/word-derangement-mapping/instruction.md) | [trajectory](trajectories/word-derangement-mapping.json) |
| 8 | `csv-json-jsonl-merger` | solved | 1.000000 | [task](tasks/csv-json-jsonl-merger/instruction.md) | [trajectory](trajectories/csv-json-jsonl-merger.json) |
| 9 | `security-breach-incident-response` | partial | 0.576667 | [task](tasks/security-breach-incident-response/instruction.md) | [trajectory](trajectories/security-breach-incident-response.json) |
| 10 | `basic-message-queue` | failed | 0.000000 | [task](tasks/basic-message-queue/instruction.md) | [trajectory](trajectories/basic-message-queue.json) |
| 11 | `multi-server-configuration` | failed | 0.000000 | [task](tasks/multi-server-configuration/instruction.md) | [trajectory](trajectories/multi-server-configuration.json) |
| 12 | `container-registry-optimization` | failed | 0.000000 | [task](tasks/container-registry-optimization/instruction.md) | [trajectory](trajectories/container-registry-optimization.json) |
| 13 | `sakila-sqlite-queries` | failed | 0.000000 | [task](tasks/sakila-sqlite-queries/instruction.md) | [trajectory](trajectories/sakila-sqlite-queries.json) |
| 14 | `network-log-normalization` | failed | 0.000000 | [task](tasks/network-log-normalization/instruction.md) | [trajectory](trajectories/network-log-normalization.json) |
| 15 | `simple-database-query-tool` | solved | 1.000000 | [task](tasks/simple-database-query-tool/instruction.md) | [trajectory](trajectories/simple-database-query-tool.json) |
| 16 | `mech-system` | infrastructure_error | 0.000000 | [task](tasks/mech-system/instruction.md) | [trajectory](trajectories/mech-system.json) |
| 17 | `rsa-jwt-token-api-redis-blacklist` | infrastructure_error | 0.000000 | [task](tasks/rsa-jwt-token-api-redis-blacklist/instruction.md) | [trajectory](trajectories/rsa-jwt-token-api-redis-blacklist.json) |
| 18 | `bash-log-processor-fix` | partial | 0.695128 | [task](tasks/bash-log-processor-fix/instruction.md) | [trajectory](trajectories/bash-log-processor-fix.json) |
| 19 | `symlink-chain-traversal` | failed | 0.000000 | [task](tasks/symlink-chain-traversal/instruction.md) | [trajectory](trajectories/symlink-chain-traversal.json) |
| 20 | `fix_async_worker_queue` | infrastructure_error | n/a | [task](tasks/fix_async_worker_queue/instruction.md) | [trajectory](trajectories/fix_async_worker_queue.json) |
| 21 | `pandas-etl` | failed | 0.000000 | [task](tasks/pandas-etl/instruction.md) | [trajectory](trajectories/pandas-etl.json) |
| 22 | `battery-charging-optimization` | failed | 0.000000 | [task](tasks/battery-charging-optimization/instruction.md) | [trajectory](trajectories/battery-charging-optimization.json) |
| 23 | `html-index-analysis` | failed | 0.000000 | [task](tasks/html-index-analysis/instruction.md) | [trajectory](trajectories/html-index-analysis.json) |
| 24 | `cryptographic-protocol-verifier` | solved | 1.000000 | [task](tasks/cryptographic-protocol-verifier/instruction.md) | [trajectory](trajectories/cryptographic-protocol-verifier.json) |
| 25 | `auth_token_race_condition` | solved | 1.000000 | [task](tasks/auth_token_race_condition/instruction.md) | [trajectory](trajectories/auth_token_race_condition.json) |
| 26 | `floor-plan-geometry` | failed | 0.000000 | [task](tasks/floor-plan-geometry/instruction.md) | [trajectory](trajectories/floor-plan-geometry.json) |
| 27 | `git-repo-forensics` | failed | 0.000000 | [task](tasks/git-repo-forensics/instruction.md) | [trajectory](trajectories/git-repo-forensics.json) |
| 28 | `log-summary` | solved | 1.000000 | [task](tasks/log-summary/instruction.md) | [trajectory](trajectories/log-summary.json) |
| 29 | `task-xxe-exploit` | infrastructure_error | 0.000000 | [task](tasks/task-xxe-exploit/instruction.md) | [trajectory](trajectories/task-xxe-exploit.json) |
| 30 | `book-portfolio-analysis` | failed | 0.000000 | [task](tasks/book-portfolio-analysis/instruction.md) | [trajectory](trajectories/book-portfolio-analysis.json) |
| 31 | `mlflow-register` | failed | 0.000000 | [task](tasks/mlflow-register/instruction.md) | [trajectory](trajectories/mlflow-register.json) |
| 32 | `sign-vector-game` | infrastructure_error | 0.000000 | [task](tasks/sign-vector-game/instruction.md) | [trajectory](trajectories/sign-vector-game.json) |
| 33 | `parking-lot-pathfinding` | failed | 0.000000 | [task](tasks/parking-lot-pathfinding/instruction.md) | [trajectory](trajectories/parking-lot-pathfinding.json) |
| 34 | `submission_a63937a5_20251224_152124` | infrastructure_error | 0.000000 | [task](tasks/submission_a63937a5_20251224_152124/instruction.md) | [trajectory](trajectories/submission_a63937a5_20251224_152124.json) |
| 35 | `supply-chain-fulfillment` | infrastructure_error | 0.000000 | [task](tasks/supply-chain-fulfillment/instruction.md) | [trajectory](trajectories/supply-chain-fulfillment.json) |
| 36 | `token-auth-websocket` | infrastructure_error | 0.000000 | [task](tasks/token-auth-websocket/instruction.md) | [trajectory](trajectories/token-auth-websocket.json) |
| 37 | `mtls-cert-rotation` | failed | 0.000000 | [task](tasks/mtls-cert-rotation/instruction.md) | [trajectory](trajectories/mtls-cert-rotation.json) |
| 38 | `build-system-task-ordering` | infrastructure_error | 0.000000 | [task](tasks/build-system-task-ordering/instruction.md) | [trajectory](trajectories/build-system-task-ordering.json) |
| 39 | `industrial-kiln-controller` | infrastructure_error | 0.000000 | [task](tasks/industrial-kiln-controller/instruction.md) | [trajectory](trajectories/industrial-kiln-controller.json) |
| 40 | `image-tile-identification` | failed | 0.000000 | [task](tasks/image-tile-identification/instruction.md) | [trajectory](trajectories/image-tile-identification.json) |
| 41 | `amuse-install` | solved | 1.000000 | [task](tasks/amuse-install/instruction.md) | [trajectory](trajectories/amuse-install.json) |
| 42 | `bandit-delayed-feedback` | failed | 0.000000 | [task](tasks/bandit-delayed-feedback/instruction.md) | [trajectory](trajectories/bandit-delayed-feedback.json) |
| 43 | `sales-data-csv-analysis` | failed | 0.000000 | [task](tasks/sales-data-csv-analysis/instruction.md) | [trajectory](trajectories/sales-data-csv-analysis.json) |
| 44 | `react-typescript-debugg` | infrastructure_error | 0.000000 | [task](tasks/react-typescript-debugg/instruction.md) | [trajectory](trajectories/react-typescript-debugg.json) |
| 45 | `python-api-rate-limit` | failed | 0.000000 | [task](tasks/python-api-rate-limit/instruction.md) | [trajectory](trajectories/python-api-rate-limit.json) |
| 46 | `monorepo-changelog-cli` | infrastructure_error | 0.000000 | [task](tasks/monorepo-changelog-cli/instruction.md) | [trajectory](trajectories/monorepo-changelog-cli.json) |
| 47 | `multi-labeller` | failed | 0.000000 | [task](tasks/multi-labeller/instruction.md) | [trajectory](trajectories/multi-labeller.json) |
| 48 | `prediction-model-evaluation` | failed | 0.000000 | [task](tasks/prediction-model-evaluation/instruction.md) | [trajectory](trajectories/prediction-model-evaluation.json) |
| 49 | `grid-pathfinding` | failed | 0.000000 | [task](tasks/grid-pathfinding/instruction.md) | [trajectory](trajectories/grid-pathfinding.json) |
| 50 | `scan-linux-persistence-artifacts` | infrastructure_error | 0.000000 | [task](tasks/scan-linux-persistence-artifacts/instruction.md) | [trajectory](trajectories/scan-linux-persistence-artifacts.json) |
