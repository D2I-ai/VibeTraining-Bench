// Chinese text for index.zh.html. Loaded after data.js; only display strings are replaced.
// Numbers, ids, benchmark / model names and the English keys used for lookups stay untouched.

(function (V) {
  const byId = (list, key = "id") => Object.fromEntries(list.map((x) => [x[key], x]));

  // Table 2 + Table 6
  const TASKS = {
    terminal: { capability: "终端操作", harness: "Terminus 2", validation: "OpenThoughts-TBLite", test: "Terminal-Bench 2.1", budget: "24 小时",
      settings: "最多 200 轮交互 · 65,536 token 上下文 · 最多输出 4,096 token · 温度 1.0 · 关闭思考模式 · 每个任务运行一次" },
    swe: { capability: "软件工程", harness: "mini-SWE-agent", validation: "150 题验证子集", test: "其余 350 题", budget: "24 小时",
      settings: "256K token 上下文 · 温度 1.0 · 开启思考模式 · 每个任务运行一次" },
    tau3: { capability: "多轮工具调用", harness: "基准自带", validation: "35 题验证子集", test: "其余 340 题", budget: "16 小时",
      settings: "最多 200 轮交互 · 256K token 上下文 · 温度 0 · 关闭思考模式 · 每个任务运行一次" },
    toolathlon: { capability: "多轮工具调用", harness: "基准自带", validation: "ToolGym", test: "Toolathlon", budget: "16 小时",
      settings: "最多 100 轮交互 · 256K token 上下文 · 最多输出 8,192 token · 温度 1.0 · 关闭思考模式 · pass@3" },
    aime: { capability: "数学推理", harness: "无（固定接口）", validation: "AIME 2025", test: "AIME 2026", budget: "16 小时",
      settings: "最多输出 16,384 token · 温度 0.6 · 开启思考模式 · average@4" },
  };
  const tasks = byId(V.TASKS);
  Object.entries(TASKS).forEach(([id, t]) => Object.assign(tasks[id], t));

  // Table 1 column headers
  const GROUPS = { "Research Loop": "研究循环", "Editable Training": "可编辑训练", "System Co-Design": "系统协同设计", "Controlled Infrastructure": "受控基础设施" };
  const LABELS = { "Trajectory Feedback": "轨迹反馈", "Iterative Revision": "迭代修订", Data: "数据", Trainer: "训练器", Harness: "推理框架", Joint: "联合设计", "Dev/Test Isolation": "开发/测试隔离", "Train/Eval Isolation": "训练/评测隔离" };
  V.PROTOCOL_COLS.forEach((c) => { c.group = GROUPS[c.group]; c.label = LABELS[c.label]; });

  // Table 3 reference rows
  Object.assign(V.REFERENCE[0], { name: "初始系统", note: "Qwen3.5-9B-Base + 初始推理框架" });
  Object.assign(V.REFERENCE[1], { name: "人工开发的 Instruct 模型", note: "Qwen3.5-9B-Instruct + 初始推理框架" });

  V.VAL_VS_TEST.forEach((r) => { r.val = r.val.replace(" (tied)", "（并列）"); });

  // Figures 5-16: [title, text] for each non-gap step, in order.
  const TRAJ = {
    "astra-swe": {
      summary: "验证轨迹暴露出 Python 解释器不匹配。在权重不变的情况下修复推理框架，解决数从 45 升到 55；随后扩大训练数据并多训一个 epoch，配合修订后的推理框架，验证集达到 68/150，隐藏测试集达到 53.7%。",
      steps: [
        ["在成功的修复示范上训练", "从 Orchard 训练集中选取 1,536 个已解决 issue 的示范，训练一个 epoch。保持原始推理框架。"],
        ["修复推理框架中的 Python 环境选择", "第一个训练模型的验证轨迹暴露出 Python 解释器不匹配。改为使用仓库已有的 Python 环境，同时保留智能体自身的解释器。权重不变时，解决数从 45 升到 55。"],
        ["测试更广的训练数据覆盖", "将 Orchard 数据从 1,536 个 issue 扩展到 4,096 个。用修订后的推理框架评测训练一个 epoch 的模型。"],
        ["测试在同一数据上追加训练", "保持相同的 4,096 个 Orchard issue 与修订后的推理框架；评测训练两个 epoch 的模型。"],
      ],
    },
    "kimi-terminal": {
      summary: "Kimi-K3 先教会模型终端交互格式，再针对错误恢复与任务完成行为进行改进。固定权重时加入的完成检查清单没有改变总分，但最终系统把它与扩充后的数据混合结合在了一起。",
      steps: [
        ["教会合法动作与显式停止", "清洗 19,652 条 TerminalTraj 示范，从基座模型开始训练。保留原有推理框架。"],
        ["在成功的恢复行为上训练", "将 9,733 条成功的 RTS 示范与 6,000 条重采样样本混合。保留格式错误的轮次作为上下文但屏蔽其损失，并从第一个检查点继续训练。"],
        ["确认完成前先核对需求", "加入完成检查清单，并在六批命令内同一批命令重复三次时发出警告。模型权重不变。"],
        ["增加每个任务的示范数量", "将 RTS 配额从每个任务三条提升到六条（含回放共 23,897 条记录）。用相同配方从第一个检查点训练；保留修订后的推理框架。"],
      ],
    },
    "astra-terminal": {
      summary: "GPT-6-Astra 在固定权重的推理框架对比与监督信号调整之间交替推进。一次解析器修复把解析错误轮次从 35 降到 23，解决的任务却更少，于是智能体恢复了先前的推理框架——把局部错误信号与任务成功区分开来。",
      steps: [
        ["让下一步动作监督与推理过程对齐", "将清洗后的轨迹展开为 16,320 条下一步动作样本，并屏蔽历史目标。训练 255 步；保留原始推理框架。"],
        ["测试完成与执行保护机制", "权重不变，验证完成标志、限制等待时长、失败后保留上下文并要求自检。解决数仍为 5。"],
        ["在恢复与完成行为上训练", "扩展到 45,248 条包含错误恢复、任务完成与有效历史目标的记录。以全新的优化器调度从第一个模型继续训练；保留修订后的推理框架。"],
        ["测试更长动作的监督", "以更低的学习率，在 20,480 条长动作、恢复与回放样本上继续训练已选模型。"],
        ["测试解析器与终端输出修复", "复用决策 3 的模型。修复 JSON 反斜杠、控制键与增量输出。解析错误轮次 35 → 23，但解决数下降。恢复决策 3 的推理框架。"],
      ],
    },
    "sol-terminal": {
      summary: "验证失败暴露出预期训练与实际执行之间的偏差：由于加载器收到的是迭代子目录，几次热启动都悄无声息地变成了随机初始化。仅修正检查点路径，验证结果就从 0/50 升到 12/50。",
      steps: [
        ["教会简洁动作与显式停止", "选取 870 条简短、合法的 TMax 轨迹，并将 batch size 从 64 降到 8。要求每次回复只包含一条命令，并识别完成标记。"],
        ["返回更新后的完成标志", "把本地更新的完成标志返回给控制循环，使识别到完成标记后能够停止执行。权重不变。"],
        ["测试低学习率下的均衡回放", "以 1e-6 学习率混合 870 条 TMax、1,024 条 TerminalTraj 与 768 条 RTS 记录。由于检查点路径错误，预期的热启动实际变成了随机初始化。"],
        ["恢复预期的父检查点", "将加载器指向检查点根目录，并确认加载了第 107 次迭代的父检查点。沿用决策 6 的数据与学习率，以及决策 3 的推理框架。"],
        ["刷新简短的成功动作", "以 1e-7 学习率在 870 条简短 TMax 轨迹上继续训练已选模型，推理框架不变。保留决策 7。"],
      ],
    },
    "ds-terminal": {
      summary: "DeepSeek-V4-Pro 从数据与执行修复转向模型合并，最终选择 0.25 / 0.75 的双模型混合；三检查点平均以及相邻系数的得分都更低。",
      steps: [
        ["训练终端技能并修复执行循环", "规范化并过滤 9,000 条示范；学习率 2e-5。在推理框架中修复 token 计数、上下文回退与重复动作反馈。"],
        ["扩充成功示范并对齐反馈", "在包含 5,000 条成功 RTS 示范的 16,000 条记录上重新训练。保留解析器修复，但去掉训练数据中不存在的修复提示。"],
        ["合并两个训练好的模型", "以 0.25 / 0.75 的系数合并决策 1 与决策 3 的权重。保留决策 3 的推理框架；不再继续训练。"],
        ["测试三个检查点的权重平均", "将决策 1 的模型与第二次训练第 199 步、第 249 步的检查点等权平均。"],
        ["测试相邻的权重配比", "以 0.30 / 0.70 的权重组合决策 1 与决策 3 的模型。保留决策 5 的系统。"],
      ],
    },
    "qwen-terminal": {
      summary: "Qwen-3.8-Max 完全通过调整数据课程来应对过早结束与长循环问题，训练配方与推理框架始终保持不变。",
      steps: [
        ["从完成的轨迹中学习终端接口", "规范化 JSON 回复，混合 8,000 条成功 RTS 与 8,000 条 TerminalTraj 示范。使用提供的配方与推理框架从基座模型训练。"],
        ["在课程中强调显式验证", "混合 5,500 条验证类 RTS、6,000 条其他 RTS 与 4,500 条 TerminalTraj 样本；TerminalTraj 限制在 40 个 assistant 轮次以内。保持 16,000 条记录、250 步与原推理框架。"],
      ],
    },
    "glm-terminal": {
      summary: "GLM-5.2 在对齐示范与终端指令之后测试数据扩充。更大的数据混合没有带来提升，最终保留了第一组模型–推理框架组合。",
      steps: [
        ["对齐示范与终端指令", "将三个数据源清洗为 6,315 条长度低于 12,000 token 的记录，并加入原始 JSON、定向编辑与自检相关的指引。用提供的配方训练一个 epoch。"],
        ["测试更大的示范混合", "扩展到 12,000 条记录：2,500 条 TerminalTraj、2,000 条 TMax 与 7,500 条成功 RTS 样本。"],
        ["用新的采样种子扩大覆盖", "用新种子测试 15,000 条记录。最终选择决策 1 的系统。"],
      ],
    },
    "kimi-swe": {
      summary: "Kimi-K3 将示范数据与所提供推理框架的格式对齐，并把每个 issue 的示范配额翻倍。尽管任务允许，它始终没有修改推理框架。",
      steps: [
        ["每个 issue 对齐一条修复示范", "把提示、工具调用、提交命令与截断后的观测转换为所提供推理框架的格式。从基座模型在 9,691 条记录（每个 issue 一条）上训练。"],
        ["测试每个 issue 两条示范", "每个 issue 两条记录（同样 9,691 个 issue，共 18,012 条）。从基座模型重新训练。新解决 20 个任务，但丢失了 17 个此前已解决的任务。"],
      ],
    },
    "sol-swe": {
      summary: "GPT-5.6-Sol 加入停止保护，把 API 调用次数的中位数从 163 降到 96.5，随后再将其与简短的回放训练结合。",
      steps: [
        ["在小规模修复示范混合上训练", "通过数据源混合与长度过滤，选取 1,024 条已解决的 Orchard 示范。从基座模型训练。"],
        ["在未见过的修复 issue 上继续训练", "以 1e-6 学习率，在 1,024 个不重叠的 issue 上继续训练一个 4,096-issue 检查点。保留原有推理框架。"],
        ["执行停滞时提交补丁", "固定决策 8 的权重，在命令重复三次或补丁连续 48 步未变时提交；拒绝超过 65,536 字节的命令。"],
        ["结合停止保护测试简短回放", "以 5e-7 学习率，在 512 条简短回放记录上对决策 8 的权重继续更新 8 步。保留停止保护。"],
        ["测试更大的简短回放切片", "回放切片从 512 条增加到 1,024 条。保留决策 11。"],
      ],
    },
    "glm-swe": {
      summary: "一次固定权重的指引修改保持了总分（52/150），却改变了哪些任务能够成功：两次运行各有 15 个独有的成功任务。",
      steps: [
        ["在简短的成功修复轨迹上训练", "按 issue 对已解决的 Orchard 记录去重，并在最短的 3,000 条合格轨迹上训练。"],
        ["扩大短轨迹子集", "使用相同配方与推理框架，从基座模型在 5,000 条短轨迹上重新训练。"],
        ["测试最小改动与自检指引", "固定决策 2 的权重，要求进行最小化的定向修改、运行已有测试并在最后检查 diff。共享成功 37 个，每次运行各有 15 个独有成功。"],
        ["测试完整的修复数据池", "在包含更长轨迹在内的全部 8,082 条记录上从头训练；恢复原始推理框架。保留决策 2。"],
      ],
    },
    "qwen-swe": {
      summary: "Qwen-3.8-Max 测试同样的 token 预算花在更广的 issue 覆盖上是否更好——结果并非如此，最终保留了第一个模型。",
      steps: [
        ["以部署格式训练成功修复", "改写提示、观测与提交命令。在 15,767 条已解决的 Orchard 记录上训练，长度上限 20,480 token。"],
        ["在 token 预算内测试更广的 issue 覆盖", "加入 Qwen 来源的轨迹，将上限提高到 28,672 token，并优先选择未见过的 issue：在 245M token 内覆盖 8,620 个 issue、共 13,965 条记录。新增 10 个成功，丢失 13 个。"],
      ],
    },
    "ds-swe": {
      summary: "DeepSeek-V4-Pro 只训练并评测了一个设计方案，使用与接口对齐的 Open-SWE 示范；它的两次验证报告是同一个系统在不同子集上的结果。",
      steps: [
        ["让接口对齐的修复数据适配训练预算", "规范化 Bash 参数并重新格式化 Open-SWE 示范。记录长度上限 48,000 token；在 550M token 内采样 17,043 条记录。每 50 次更新保存一次。"],
      ],
    },
  };
  V.TRAJECTORIES.forEach((t) => {
    const zh = TRAJ[t.id];
    t.summary = zh.summary;
    t.steps.filter((s) => !s.gap).forEach((s, i) => { [s.title, s.text] = zh.steps[i]; });
  });
})(window.VTB);
