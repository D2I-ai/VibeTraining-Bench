// All numbers below are transcribed from the VibeTraining-Bench paper.
// Table 3 (hidden-test), Table 1 (protocol), Table 2/6 (tasks), Table 4 (ablation),
// Figure 4 (design counts) and Figures 5-16 (research trajectories).

window.VTB = (function () {
  const AGENTS = [
    { id: "astra", name: "GPT-6-Astra", short: "Astra", framework: "Codex · xhigh", color: "#7b52ab" },
    { id: "sol", name: "GPT-5.6-Sol", short: "Sol", framework: "Codex · xhigh", color: "#5f7481" },
    { id: "kimi", name: "Kimi-K3", short: "Kimi", framework: "Claude Code", color: "#2f7fdc" },
    { id: "glm", name: "GLM-5.2", short: "GLM", framework: "Claude Code", color: "#1f8a3a" },
    { id: "qwen", name: "Qwen-3.8-Max", short: "Qwen", framework: "Claude Code", color: "#e5739b" },
    { id: "ds", name: "DeepSeek-V4-Pro", short: "DS", framework: "Claude Code", color: "#f0a020" },
  ];

  // Column order follows Table 3.
  const TASK_COLS = [
    { id: "swe", label: "SWE-bench Verified", short: "SWE" },
    { id: "terminal", label: "Terminal-Bench 2.1", short: "Terminal" },
    { id: "tau3", label: "τ³-Bench", short: "τ³" },
    { id: "toolathlon", label: "Toolathlon", short: "Toolathlon" },
    { id: "aime", label: "AIME 2026", short: "AIME" },
  ];

  // Table 3: hidden-test performance (%) of the selected target agent systems.
  const REFERENCE = [
    { id: "initial", name: "Initial system", note: "Qwen3.5-9B-Base + initial harness", avg: 22.7, swe: 32.9, terminal: 8.9, tau3: 20.3, toolathlon: 4.5, aime: 46.7, hours: null },
    { id: "instruct", name: "Human-developed instruct model", note: "Qwen3.5-9B-Instruct + initial harness", avg: 47.3, swe: 55.4, terminal: 20.2, tau3: 64.1, toolathlon: 21.6, aime: 75.0, hours: null },
  ];
  const RESULTS = [
    { agent: "astra", avg: 37.1, swe: 53.7, terminal: 13.5, tau3: 49.7, toolathlon: 7.9, aime: 60.8, hours: 85.9 },
    { agent: "sol", avg: 31.0, swe: 42.9, terminal: 8.9, tau3: 42.1, toolathlon: 10.2, aime: 50.8, hours: 64.0 },
    { agent: "kimi", avg: 38.3, swe: 51.7, terminal: 22.4, tau3: 52.1, toolathlon: 10.2, aime: 55.0, hours: 73.0 },
    { agent: "glm", avg: 31.4, swe: 40.6, terminal: 10.1, tau3: 36.2, toolathlon: 6.8, aime: 63.3, hours: 56.9 },
    { agent: "qwen", avg: 35.6, swe: 48.0, terminal: 17.9, tau3: 47.4, toolathlon: 9.1, aime: 55.8, hours: 81.8 },
    { agent: "ds", avg: 34.2, swe: 41.4, terminal: 13.5, tau3: 47.9, toolathlon: 12.5, aime: 55.8, hours: 87.0 },
  ];

  // Table 2 + Table 6 + Appendix C.3.
  const TASKS = [
    {
      id: "terminal", name: "Terminal-Bench 2.1", capability: "Terminal operation", icon: "terminal",
      harness: "Terminus 2", editable: ["D", "T", "H"],
      validation: "OpenThoughts-TBLite", test: "Terminal-Bench 2.1", budget: "24 h",
      settings: "Max 200 agent turns · 65,536-token context · 4,096 max output tokens · temperature 1.0 · thinking disabled · one run per task",
      data: ["TMax-SFT-16.5K", "Recursive Task Synthesis Trajectories", "TerminalTraj"],
    },
    {
      id: "swe", name: "SWE-bench Verified", capability: "Software engineering", icon: "code",
      harness: "mini-SWE-agent", editable: ["D", "T", "H"],
      validation: "150-task validation subset", test: "Remaining 350 tasks", budget: "24 h",
      settings: "256K-token context · temperature 1.0 · thinking enabled · one run per task",
      data: ["Orchard", "SWE-Dev-Train", "Nebius SWE-rebench", "NVIDIA Open-SWE-Traces"],
    },
    {
      id: "tau3", name: "τ³-Bench", capability: "Multi-turn tool use", icon: "chat",
      harness: "Benchmark-provided", editable: ["D", "T"],
      validation: "35-task validation subset", test: "Remaining 340 tasks", budget: "16 h",
      settings: "Max 200 agent turns · 256K-token context · temperature 0 · thinking disabled · one run per task",
      data: ["AReaL-tau2-data", "Nemotron-SFT-Agentic-v2", "Simia-Tau-SFT"],
    },
    {
      id: "toolathlon", name: "Toolathlon", capability: "Multi-turn tool use", icon: "tool",
      harness: "Benchmark-provided", editable: ["D", "T"],
      validation: "ToolGym", test: "Toolathlon", budget: "16 h",
      settings: "Max 100 agent turns · 256K-token context · 8,192 max output tokens · temperature 1.0 · thinking disabled · pass@3",
      data: ["DIVE-SFT-20K"],
    },
    {
      id: "aime", name: "AIME", capability: "Mathematical reasoning", icon: "math",
      harness: "None (fixed interface)", editable: ["D", "T"],
      validation: "AIME 2025", test: "AIME 2026", budget: "16 h",
      settings: "16,384 max output tokens · temperature 0.6 · thinking enabled · average@4",
      data: ["Nemotron-SFT-Math-v4", "OpenR1-Math-220k", "OpenThoughts3-1.2M"],
    },
  ];

  // Table 1: Y = enforced, P = partial / insufficiently documented, N = unsupported, "-" = n/a.
  const PROTOCOL_COLS = [
    { group: "Research Loop", label: "Trajectory Feedback" },
    { group: "Research Loop", label: "Iterative Revision" },
    { group: "Editable Training", label: "Data" },
    { group: "Editable Training", label: "Trainer" },
    { group: "System Co-Design", label: "Harness" },
    { group: "System Co-Design", label: "Joint" },
    { group: "Controlled Infrastructure", label: "Dev/Test Isolation" },
    { group: "Controlled Infrastructure", label: "Train/Eval Isolation" },
  ];
  const PROTOCOL = [
    { name: "MLS-Bench", v: "PYNYNNPN" },
    { name: "PostTrainBench", v: "YNYYNNNN" },
    { name: "Curation-Bench", v: "YYYNNNNP" },
    { name: "RSIBench-Data", v: "YYYNNNNY" },
    { name: "Agent² RL-Bench", v: "NYYYNNNY" },
    { name: "AI4AI-Bench", v: "PYNYNNYY" },
    { name: "HarnessOpt-Bench", v: "YYNNYNY-" },
    { name: "Evo-Bench", v: "YYNNYNY-" },
    { name: "VibeTraining-Bench", v: "YYYYYYYY", ours: true },
  ];

  // Figure 4: number of designs involving Data / Trainer / Harness. total = distinct designs.
  const DESIGNS = {
    aime: { label: "AIME 2026", rows: { astra: [8, 15, 0, 17], sol: [7, 13, 0, 17], kimi: [3, 4, 0, 5], glm: [5, 11, 0, 12], qwen: [5, 1, 0, 6], ds: [5, 2, 0, 6] }, harness: false },
    terminal: { label: "Terminal-Bench 2.1", rows: { astra: [3, 2, 2, 5], sol: [6, 9, 5, 11], kimi: [3, 1, 1, 4], glm: [3, 0, 1, 3], qwen: [2, 0, 0, 2], ds: [2, 6, 3, 8] }, harness: true },
    swe: { label: "SWE-bench Verified", rows: { astra: [3, 5, 4, 9], sol: [11, 3, 2, 13], kimi: [2, 0, 0, 2], glm: [3, 0, 1, 4], qwen: [2, 0, 0, 2], ds: [1, 0, 0, 1] }, harness: true },
    tau3: { label: "τ³-Bench", rows: { astra: [7, 9, 0, 9], sol: [7, 7, 0, 7], kimi: [3, 1, 0, 3], glm: [3, 3, 0, 4], qwen: [4, 2, 0, 5], ds: [3, 8, 0, 8] }, harness: false },
    toolathlon: { label: "Toolathlon", rows: { astra: [5, 8, 0, 12], sol: [6, 5, 0, 6], kimi: [1, 2, 0, 2], glm: [2, 1, 0, 3], qwen: [3, 0, 0, 3], ds: [1, 2, 0, 2] }, harness: false },
  };

  // Section 4.3: validation leadership vs. hidden-test leadership.
  const VAL_VS_TEST = [
    { task: "AIME 2026", val: "GPT-5.6-Sol", test: "GLM-5.2", testScore: 63.3, transfer: false },
    { task: "Terminal-Bench 2.1", val: "GPT-6-Astra", test: "Kimi-K3", testScore: 22.4, transfer: false, note: "Astra: 26% validation → 13.5% hidden test" },
    { task: "SWE-bench Verified", val: "GPT-6-Astra", test: "GPT-6-Astra", testScore: 53.7, transfer: true },
    { task: "τ³-Bench", val: "GPT-6-Astra", test: "Kimi-K3", testScore: 52.1, transfer: false, note: "Astra: 49.7% hidden test" },
    { task: "Toolathlon", val: "GPT-6-Astra (tied)", test: "DeepSeek-V4-Pro", testScore: 12.5, transfer: false },
  ];

  // Table 4: Kimi-K3 component ablation (hidden test, %).
  const ABLATION = [
    { setting: "Initial system", swe: 32.9, terminal: 8.9 },
    { setting: "H only", swe: 38.0, terminal: 8.9 },
    { setting: "D/T only", swe: null, terminal: 13.5 },
    { setting: "D/T/H", swe: 51.7, terminal: 22.4, note: "Kimi-K3 did not edit the harness in its SWE D/T/H run" },
  ];

  // Figures 5-16: selected decisions from research trajectories.
  // c = components touched; v = [solved, graded]; sub = subset validation; sel = selected system.
  const TRAJECTORIES = [
    {
      id: "astra-swe", agent: "astra", task: "SWE-bench Verified", hidden: 53.7, fig: "Figure 5",
      summary: "Validation traces expose a Python interpreter mismatch. Fixing the harness at fixed weights lifts solves from 45 to 55; broader data plus a second epoch, evaluated with the revised harness, reaches 68/150 and 53.7% on the hidden test.",
      steps: [
        { n: 1, c: ["D"], title: "Train on successful repair demonstrations", text: "Select demonstrations for 1,536 resolved issues from the Orchard training dataset and train for one epoch. Keep the original harness.", v: [45, 150] },
        { gap: true },
        { n: 3, c: ["H"], title: "Fix Python environment selection in the harness", text: "Validation traces from the first trained model reveal a Python interpreter mismatch. Use the repository's existing Python environment while preserving the agent's own interpreter. With weights fixed, solved tasks rise from 45 to 55.", v: [55, 150] },
        { gap: true },
        { n: 7, c: ["D"], title: "Test broader training-data coverage", text: "Expand Orchard data from 1,536 to 4,096 issues. Evaluate the one-epoch model with the revised harness.", v: [54, 150] },
        { gap: true },
        { n: 9, c: ["T"], title: "Test additional training on the same data", text: "Keep the same 4,096 Orchard issues and revised harness; evaluate the model trained for two epochs.", v: [68, 150], sel: true },
      ],
    },
    {
      id: "kimi-terminal", agent: "kimi", task: "Terminal-Bench 2.1", hidden: 22.4, fig: "Figure 6",
      summary: "Kimi-K3 first teaches the terminal interaction format, then targets recovery and completion behaviour. A fixed-weight completion checklist leaves the total unchanged, but the final system combines it with an expanded data mixture.",
      steps: [
        { n: 1, c: ["D"], title: "Teach valid actions and explicit stopping", text: "Clean 19,652 TerminalTraj demonstrations and train from the base model. Keep the supplied harness.", v: [4, 50] },
        { n: 2, c: ["D", "T"], title: "Train on successful recovery behavior", text: "Mix 9,733 successful RTS demonstrations with 6,000 resampled examples. Keep malformed turns as context, mask their loss, and continue the first checkpoint.", v: [10, 50] },
        { n: 3, c: ["H"], title: "Check requirements before confirming completion", text: "Add a completion checklist and warn when a command batch repeats three times in six batches. Model weights fixed.", v: [10, 50] },
        { n: 4, c: ["D"], title: "Increase demonstrations per task", text: "Raise the RTS quota from three to six traces per task (23,897 records with replay). Train from the first checkpoint with the same recipe; retain the revised harness.", v: [12, 50], sel: true },
      ],
    },
    {
      id: "astra-terminal", agent: "astra", task: "Terminal-Bench 2.1", hidden: 13.5, fig: "Figure 11",
      summary: "GPT-6-Astra alternates fixed-weight harness comparisons with supervision changes. A parser repair cuts parser-error turns from 35 to 23 yet solves fewer tasks, so the agent restores the earlier harness — separating local error signals from task success.",
      steps: [
        { n: 1, c: ["D"], title: "Match next-action supervision to inference", text: "Expand cleaned trajectories into 16,320 next-action examples and mask historical targets. Train 255 steps; retain the original harness.", v: [5, 50] },
        { n: 2, c: ["H"], title: "Test completion and execution safeguards", text: "With weights fixed, validate completion flags, bound waits, preserve context after failures, and request verification. Solved count stays at five.", v: [5, 50] },
        { n: 3, c: ["D", "T"], title: "Train on recovery and completion behavior", text: "Expand to 45,248 records with recovery, completion and valid historical targets. Continue the first model with a fresh optimizer schedule; retain the revised harness.", v: [13, 50], sel: true },
        { n: 4, c: ["D", "T"], title: "Test longer-action supervision", text: "Continue the selected model on 20,480 long-action, recovery and replay examples at a lower learning rate.", v: [8, 50] },
        { n: 5, c: ["H"], title: "Test parser and terminal-output repairs", text: "Reuse the Decision 3 model. Repair JSON backslashes, control keys and incremental output. Parser-error turns fall 35 → 23, but solves drop. Restore the Decision 3 harness.", v: [9, 50] },
      ],
    },
    {
      id: "sol-terminal", agent: "sol", task: "Terminal-Bench 2.1", hidden: 8.9, fig: "Figure 8",
      summary: "Validation failures expose a mismatch between intended and executed training: several warm starts silently initialise randomly because the loader receives an iteration directory. Fixing the checkpoint path alone raises validation from 0/50 to 12/50.",
      steps: [
        { n: 1, c: ["D", "T", "H"], title: "Teach concise actions and explicit stopping", text: "Select 870 short, valid TMax traces and reduce batch size from 64 to 8. Request one command per response and recognise completion markers.", v: [7, 50] },
        { n: 3, c: ["H"], title: "Return the updated completion flag", text: "Return the locally updated completion flag to the control loop so marker recognition stops execution. Weights fixed.", v: [7, 50] },
        { gap: true },
        { n: 6, c: ["D", "T"], title: "Test balanced replay at a lower learning rate", text: "Mix 870 TMax, 1,024 TerminalTraj and 768 RTS records at 1e-6. The intended warm start initialises randomly because its checkpoint path is wrong.", v: [0, 50] },
        { n: 7, c: ["T"], title: "Restore the intended parent checkpoint", text: "Point the loader to the checkpoint root and verify parent iteration 107 loads. Keep Decision 6 data and LR and the Decision 3 harness.", v: [12, 50], sel: true },
        { gap: true },
        { n: 11, c: ["D", "T"], title: "Refresh short successful actions", text: "Continue the selected model on 870 short TMax traces at 1e-7 with the same harness. Retain Decision 7.", v: [5, 50] },
      ],
    },
    {
      id: "ds-terminal", agent: "ds", task: "Terminal-Bench 2.1", hidden: 13.5, fig: "Figure 10",
      summary: "DeepSeek-V4-Pro moves from data and execution repairs to model merging, selecting a 0.25 / 0.75 two-model mixture; three-checkpoint averaging and neighbouring coefficients score lower.",
      steps: [
        { n: 1, c: ["D", "T", "H"], title: "Train terminal skills and repair execution loops", text: "Canonicalise and filter 9,000 demonstrations; LR 2e-5. Repair token counting, context fallback and repeated-action feedback in the harness.", v: [10, 50] },
        { gap: true },
        { n: 3, c: ["D", "H"], title: "Expand successful demonstrations and align feedback", text: "Retrain on 16,000 records including 5,000 successful RTS demonstrations. Keep parser repairs but suppress repair notices absent from training.", v: [10, 50] },
        { gap: true },
        { n: 5, c: ["T"], title: "Merge the two trained models", text: "Merge weights from Decisions 1 and 3 with coefficients 0.25 / 0.75. Keep the Decision 3 harness; no further training.", v: [12, 50], sel: true },
        { n: 6, c: ["T"], title: "Test weight averaging across three checkpoints", text: "Equally average the Decision 1 model with the step-199 and step-249 checkpoints of the second run.", v: [10, 50] },
        { gap: true },
        { n: 8, c: ["T"], title: "Test a neighboring weight mixture", text: "Combine the Decision 1 and 3 models with weights 0.30 / 0.70. Retain the Decision 5 system.", v: [5, 50] },
      ],
    },
    {
      id: "qwen-terminal", agent: "qwen", task: "Terminal-Bench 2.1", hidden: 17.9, fig: "Figure 7",
      summary: "Qwen-3.8-Max responds to premature completion and long loops purely through the data curriculum, keeping the training recipe and harness fixed.",
      steps: [
        { n: 1, c: ["D"], title: "Teach the terminal interface from completed trajectories", text: "Normalise JSON responses and mix 8,000 successful RTS with 8,000 TerminalTraj demonstrations. Train from the base with the supplied recipe and harness.", v: [5, 50] },
        { n: 2, c: ["D"], title: "Emphasize explicit verification in the curriculum", text: "Mix 5,500 verification RTS, 6,000 other RTS and 4,500 TerminalTraj examples; cap TerminalTraj at 40 assistant turns. Keep 16,000 records, 250 steps and the harness.", v: [9, 50], sel: true },
      ],
    },
    {
      id: "glm-terminal", agent: "glm", task: "Terminal-Bench 2.1", hidden: 10.1, fig: "Figure 9",
      summary: "GLM-5.2 tests data expansion after aligning demonstrations and terminal instructions. Larger mixtures do not help, and the first model–harness pair is retained.",
      steps: [
        { n: 1, c: ["D", "H"], title: "Align demonstrations and terminal instructions", text: "Clean three sources to 6,315 records below 12,000 tokens and add raw-JSON, targeted-edit and verification guidance. One epoch with the supplied recipe.", v: [8, 50], sel: true },
        { n: 2, c: ["D"], title: "Test a larger demonstration mixture", text: "Expand to 12,000 records: 2,500 TerminalTraj, 2,000 TMax and 7,500 successful RTS examples.", v: [8, 50] },
        { n: 3, c: ["D"], title: "Expand coverage with a new sampling seed", text: "Test 15,000 records with a new seed. Select the Decision 1 system.", v: [5, 50] },
      ],
    },
    {
      id: "kimi-swe", agent: "kimi", task: "SWE-bench Verified", hidden: 51.7, fig: "Figure 14",
      summary: "Kimi-K3 aligns demonstrations to the supplied harness format and doubles the per-issue quota. It never edits the harness, even though the task permits it.",
      steps: [
        { n: 1, c: ["D"], title: "Align one repair demonstration per issue", text: "Translate prompts, tool calls, submission commands and truncated observations into the supplied harness format. Train from the base on 9,691 records, one per issue.", v: [60, 150] },
        { n: 2, c: ["D"], title: "Test two demonstrations per issue", text: "Two records per issue (18,012 records over the same 9,691 issues). Retrain from the base. Solves 20 new tasks but loses 17 previously solved ones.", v: [63, 150], sel: true },
      ],
    },
    {
      id: "sol-swe", agent: "sol", task: "SWE-bench Verified", hidden: 42.9, fig: "Figure 15",
      summary: "GPT-5.6-Sol adds a stopping guard that cuts median API calls from 163 to 96.5, then combines it with short replay training.",
      steps: [
        { n: 1, c: ["D"], title: "Train on a small repair-demonstration mixture", text: "Select 1,024 resolved Orchard demonstrations using source mixing and length filtering. Train from the base model.", v: [6, 30], sub: true },
        { gap: true },
        { n: 8, c: ["D", "T"], title: "Continue training on unseen repair issues", text: "Continue a 4,096-issue checkpoint on 1,024 disjoint issues at LR 1e-6. Keep the supplied harness.", v: [8, 30], sub: true },
        { gap: true },
        { n: 10, c: ["H"], title: "Submit the patch when execution stalls", text: "With Decision 8 weights fixed, submit after three repeated commands or 48 unchanged-patch steps; reject commands longer than 65,536 bytes.", v: [45, 150] },
        { n: 11, c: ["D", "T"], title: "Test short replay with the stopping guard", text: "Continue Decision 8 weights on 512 short replay records for eight updates at 5e-7. Retain the guard.", v: [47, 150], sel: true },
        { n: 12, c: ["D"], title: "Test a larger short-replay slice", text: "Increase the replay slice from 512 to 1,024 records. Retain Decision 11.", v: [8, 30], sub: true },
      ],
    },
    {
      id: "glm-swe", agent: "glm", task: "SWE-bench Verified", hidden: 40.6, fig: "Figure 16",
      summary: "A fixed-weight guidance change preserves the aggregate score (52/150) while reshuffling which tasks succeed: each run has 15 unique successes.",
      steps: [
        { n: 1, c: ["D"], title: "Train on short successful repair trajectories", text: "Deduplicate resolved Orchard records by issue and train on the shortest 3,000 eligible trajectories.", v: [45, 150] },
        { n: 2, c: ["D"], title: "Expand the short-trajectory subset", text: "Train afresh from the base on 5,000 short trajectories with the same recipe and harness.", v: [52, 150], sel: true },
        { n: 3, c: ["H"], title: "Test minimal-edit and verification guidance", text: "With Decision 2 weights fixed, request minimal targeted edits, existing tests and a final diff review. 37 shared successes, 15 unique per run.", v: [52, 150] },
        { n: 4, c: ["D"], title: "Test the complete repair-data pool", text: "Train afresh on all 8,082 records including longer trajectories; restore the original harness. Retain Decision 2.", v: [50, 150] },
      ],
    },
    {
      id: "qwen-swe", agent: "qwen", task: "SWE-bench Verified", hidden: 48.0, fig: "Figure 12",
      summary: "Qwen-3.8-Max tests whether the same token budget is better spent on broader issue coverage — it is not, and the first model is kept.",
      steps: [
        { n: 1, c: ["D"], title: "Train successful repairs in the deployed format", text: "Rewrite prompts, observations and submission commands. Train on 15,767 resolved Orchard records capped at 20,480 tokens.", v: [59, 150], sel: true },
        { n: 2, c: ["D"], title: "Test broader issue coverage within a token budget", text: "Add Qwen-source traces, raise the cap to 28,672 tokens and prioritise unseen issues: 13,965 records over 8,620 issues within 245M tokens. Ten gains, thirteen losses.", v: [56, 150] },
      ],
    },
    {
      id: "ds-swe", agent: "ds", task: "SWE-bench Verified", hidden: 41.4, fig: "Figure 13",
      summary: "DeepSeek-V4-Pro trains one evaluated design on interface-aligned Open-SWE demonstrations; its two validation reports are the same system on different subsets.",
      steps: [
        { n: 1, c: ["D"], title: "Fit interface-aligned repair data to the training budget", text: "Normalise Bash arguments and reformat Open-SWE demonstrations. Cap records at 48,000 tokens; sample 17,043 records within 550M tokens. Save every 50 updates.", v: [23, 60], sub: true, sel: true },
      ],
    },
  ];

  return { AGENTS, TASK_COLS, REFERENCE, RESULTS, TASKS, PROTOCOL_COLS, PROTOCOL, DESIGNS, VAL_VS_TEST, ABLATION, TRAJECTORIES };
})();
