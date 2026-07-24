import * as fs from "node:fs/promises";
import * as path from "node:path";
import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";

type Host = { enabled?: boolean; backend?: string; ssh_alias?: string; profile?: string };
type Inventory = { version?: number; hosts?: Record<string, Host> };
type CostRole = { model?: string | string[]; thinking_level?: string; prewalk_target?: string };
type CostProfile = {
  description?: string;
  roles?: Record<string, CostRole>;
  prewalk?: { default?: string; situations?: Record<string, string> };
};
type CostPolicy = {
  version?: number;
  active_profile?: string;
  profiles?: Record<string, CostProfile>;
  developer_overrides?: {
    roles?: Record<string, CostRole>;
    prewalk_situations?: Record<string, string>;
  };
};

const ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$/;

async function exists(file: string): Promise<boolean> {
  try { await fs.access(file); return true; } catch { return false; }
}

async function send(pi: ExtensionAPI, text: string): Promise<void> {
  await pi.sendUserMessage(text);
}

function words(args: string): string[] {
  return args.trim().split(/\s+/).filter(Boolean);
}

function validId(value: string, label: string): string {
  if (!ID.test(value)) throw new Error(`${label} must use letters, numbers, '.', '_' or '-' and start alphanumerically.`);
  return value;
}

async function seed(cwd: string): Promise<string[]> {
  const root = path.join(cwd, ".omp", "gpu-lab");
  const templateRoot = path.join(root, "templates");
  const created: string[] = [];
  await fs.mkdir(path.join(root, "results"), { recursive: true });
  await fs.mkdir(path.join(root, "generated"), { recursive: true });
  for (const name of ["hosts.json", "project.json", "cost-policy.json", "exploration-policy.json", "omp-config.yml", "ssh-config.fragment"]) {
    const source = path.join(templateRoot, name);
    const target = path.join(root, name);
    if (!(await exists(source))) throw new Error(`Missing installed GPU-lab template: ${source}`);
    if (!(await exists(target))) {
      await fs.copyFile(source, target);
      created.push(path.relative(cwd, target));
    }
  }
  return created;
}

async function initializeAgentWiki(cwd: string): Promise<string[]> {
  const policyFile = path.join(cwd, ".omp", "gpu-lab", "exploration-policy.json");
  const policy = JSON.parse(await fs.readFile(policyFile, "utf8")) as { defaults?: { docs_root?: string } };
  const docsRoot = policy.defaults?.docs_root ?? "docs/agent-wiki";
  const root = path.resolve(cwd, docsRoot);
  const docsBase = path.resolve(cwd, "docs");
  if (root !== docsBase && !root.startsWith(`${docsBase}${path.sep}`)) {
    throw new Error("Configured docs_root must remain under the project docs/ directory.");
  }
  const created: string[] = [];
  await fs.mkdir(path.join(root, "raw"), { recursive: true });
  await fs.mkdir(path.join(root, "wiki"), { recursive: true });
  const files: Array<[string, string]> = [
    ["index.md", "# Project Agent Wiki\n\nDurable project knowledge maintained by `specflow-doc-librarian`.\n\n## Topics\n\n_No compiled pages yet._\n"],
    ["log.md", "# Agent Wiki Operation Log\n\nAppend-only record of knowledge-base updates.\n"],
  ];
  for (const [name, content] of files) {
    const file = path.join(root, name);
    if (!(await exists(file))) { await fs.writeFile(file, content, "utf8"); created.push(path.relative(cwd, file)); }
  }
  return created;
}

async function inventory(cwd: string): Promise<Inventory> {
  const file = path.join(cwd, ".omp", "gpu-lab", "hosts.json");
  const data = JSON.parse(await fs.readFile(file, "utf8")) as Inventory;
  if (data.version !== 1 || !data.hosts) throw new Error(`${file} is not a version-1 GPU-lab inventory.`);
  return data;
}

async function costPolicy(cwd: string): Promise<{ file: string; data: CostPolicy }> {
  const file = path.join(cwd, ".omp", "gpu-lab", "cost-policy.json");
  const data = JSON.parse(await fs.readFile(file, "utf8")) as CostPolicy;
  if (data.version !== 1 || !data.profiles || !data.active_profile || !data.profiles[data.active_profile]) {
    throw new Error(`${file} is not a valid version-1 cost policy.`);
  }
  return { file, data };
}

async function saveCostPolicy(file: string, data: CostPolicy): Promise<void> {
  await fs.writeFile(file, `${JSON.stringify(data, null, 2)}\n`, "utf8");
}

function renderModel(value: string | string[] | undefined): string {
  if (Array.isArray(value)) return value.join(" -> ");
  return value ?? "inherit";
}

function effectivePrewalk(data: CostPolicy, situation: string): string {
  const override = data.developer_overrides?.prewalk_situations?.[situation];
  if (override) return override;
  const profile = data.profiles?.[data.active_profile ?? ""];
  return profile?.prewalk?.situations?.[situation] ?? profile?.prewalk?.default ?? "auto";
}

function effectiveCostRole(data: CostPolicy, role: string): CostRole {
  const base = data.profiles?.[data.active_profile ?? ""]?.roles?.[role] ?? {};
  const override = data.developer_overrides?.roles?.[role] ?? {};
  return { ...base, ...override };
}

function withThinking(value: string | string[] | undefined, thinking: string | undefined): string {
  const add = (selector: string): string => {
    if (!thinking || thinking === "auto") return selector;
    return `${selector.replace(/:(off|minimal|low|medium|high|xhigh|max)$/, "")}:${thinking}`;
  };
  if (Array.isArray(value)) return value.map(add).join(" -> ");
  return value ? add(value) : "inherit";
}

async function syncPrewalkOverlay(cwd: string, data: CostPolicy): Promise<void> {
  const file = path.join(cwd, ".omp", "gpu-lab", "omp-config.yml");
  if (!(await exists(file))) return;
  const target = effectiveCostRole(data, "implementer_prewalk").prewalk_target ?? "@smol";
  let lines = (await fs.readFile(file, "utf8")).split(/\r?\n/);
  const taskStart = lines.findIndex((line) => line === "task:");
  if (taskStart < 0) throw new Error(`${file} has no top-level task section.`);
  let taskEnd = lines.findIndex((line, index) => index > taskStart && line.length > 0 && !line.startsWith(" ") && !line.startsWith("#"));
  if (taskEnd < 0) taskEnd = lines.length;
  const prewalkStart = lines.findIndex((line, index) => index > taskStart && index < taskEnd && line.trim() === "agentPrewalk:" && line.startsWith("  "));
  const targetLine = `    specflow-implementer-prewalk: ${JSON.stringify(target)}`;
  if (prewalkStart < 0) {
    const block = [
      "",
      "  # Managed by gpu-lab-cost: only the editing variant is armed.",
      "  agentPrewalk:",
      '    specflow-implementer: "off"',
      targetLine,
      '    specflow-reviewer: "off"',
      '    specflow-tester: "off"',
      '    specflow-final-reviewer: "off"',
      '    specflow-gpu-cuda-validator: "off"',
      '    specflow-gpu-rocm-validator: "off"',
      '    specflow-gpu-portability-reviewer: "off"',
      '    specflow-gpu-performance-reviewer: "off"',
      '    specflow-gpu-debugger: "off"',
      '    scout: "off"',
      '    librarian: "off"',
      '    specflow-doc-librarian: "off"',
    ];
    lines.splice(taskEnd, 0, ...block);
  } else {
    let sectionEnd = taskEnd;
    for (let i = prewalkStart + 1; i < taskEnd; i += 1) {
      if (lines[i] && lines[i].length - lines[i].trimStart().length <= 2) { sectionEnd = i; break; }
    }
    const managed: Record<string, string> = {
      "specflow-implementer": '"off"',
      "specflow-implementer-prewalk": JSON.stringify(target),
      "specflow-reviewer": '"off"',
      "specflow-tester": '"off"',
      "specflow-final-reviewer": '"off"',
      "specflow-gpu-cuda-validator": '"off"',
      "specflow-gpu-rocm-validator": '"off"',
      "specflow-gpu-portability-reviewer": '"off"',
      "specflow-gpu-performance-reviewer": '"off"',
      "specflow-gpu-debugger": '"off"',
      scout: '"off"',
      librarian: '"off"',
      "specflow-doc-librarian": '"off"',
    };
    for (const [agent, rendered] of Object.entries(managed)) {
      const prefix = `    ${agent}:`;
      const current = lines.findIndex((line, index) => index > prewalkStart && index < sectionEnd && line.startsWith(prefix));
      const line = `    ${agent}: ${rendered}`;
      if (current < 0) { lines.splice(sectionEnd, 0, line); sectionEnd += 1; } else lines[current] = line;
    }
  }
  await fs.writeFile(file, `${lines.join("\n").replace(/\n+$/, "")}\n`, "utf8");
}

export default function gpuLab(pi: ExtensionAPI) {
  pi.registerCommand("gpu-lab-init", {
    description: "Seed project GPU-lab inventory and runtime configuration without overwriting existing files",
    handler: async (_args, ctx) => {
      try {
        const created = await seed(ctx.cwd);
        const wikiCreated = await initializeAgentWiki(ctx.cwd);
        const allCreated = [...created, ...wikiCreated];
        ctx.ui.notify(allCreated.length ? `GPU-lab created: ${allCreated.join(", ")}` : "GPU-lab already initialized; existing configuration preserved.", "info");
        await send(pi, [
          "Load and follow the specflow-gpu-orchestrator skill for environment setup.",
          "Edit `.omp/gpu-lab/hosts.json`, enable the real CUDA/ROCm hosts, and copy `.omp/gpu-lab/ssh-config.fragment` into `~/.ssh/config` with real addresses and keys.",
          "Then run `.omp/gpu-lab/bin/gpu-lab-configure` and `.omp/gpu-lab/bin/gpu-labctl doctor --all`.",
          "Do not modify existing host credentials or SDK installations without explicit developer direction.",
        ].join("\n"));
      } catch (error) {
        ctx.ui.notify(error instanceof Error ? error.message : String(error), "error");
      }
    },
  });

  pi.registerCommand("gpu-lab-status", {
    description: "Show enabled CUDA/ROCm hosts and generated debugger state",
    handler: async (_args, ctx) => {
      try {
        const data = await inventory(ctx.cwd);
        const rows = Object.entries(data.hosts ?? {}).map(([id, host]) =>
          `${id}: enabled=${Boolean(host.enabled)} backend=${host.backend ?? "?"} ssh=${host.ssh_alias ?? "?"} profile=${host.profile ?? "?"}`,
        );
        const dap = await exists(path.join(ctx.cwd, ".omp", "dap.json"));
        const mcp = await exists(path.join(ctx.cwd, ".omp", "mcp.json"));
        ctx.ui.notify(`${rows.join(" | ") || "No hosts configured"} | dap=${dap} | mcp=${mcp}`, "info");
      } catch (error) {
        ctx.ui.notify(error instanceof Error ? error.message : String(error), "error");
      }
    },
  });

  pi.registerCommand("gpu-lab-doctor", {
    description: "Ask the agent to run deterministic laptop and remote GPU-host diagnostics",
    handler: async (args, ctx) => {
      const target = args.trim() || "--all";
      await send(pi, [
        "Load and follow the specflow-gpu-testing skill.",
        `Run \`.omp/gpu-lab/bin/gpu-labctl doctor ${target}\` from \`${ctx.cwd}\`.`,
        "Do not install packages or alter drivers. Classify missing tools and give exact setup-document references.",
      ].join("\n"));
    },
  });


  pi.registerCommand("gpu-cost-status", {
    description: "Show the active SpecFlow subagent model-routing and prewalk policy",
    handler: async (_args, ctx) => {
      try {
        const { data } = await costPolicy(ctx.cwd);
        const profile = data.profiles?.[data.active_profile ?? ""];
        const roles = Object.keys(profile?.roles ?? {}).map((role) => {
          const value = effectiveCostRole(data, role);
          return `${role}=${withThinking(value.model, value.thinking_level)}`;
        });
        const prewalk = ["mechanical", "repository-heavy-small-edit", "normal", "cross-cutting", "high-risk", "debugging"]
          .map((item) => `${item}:${effectivePrewalk(data, item)}`);
        ctx.ui.notify(`profile=${data.active_profile} | ${roles.join(" | ")} | prewalk ${prewalk.join(", ")}`, "info");
      } catch (error) {
        ctx.ui.notify(error instanceof Error ? error.message : String(error), "error");
      }
    },
  });

  pi.registerCommand("gpu-cost-profile", {
    description: "Select the economy, balanced, quality, or custom cost-policy profile",
    handler: async (args, ctx) => {
      try {
        const name = args.trim();
        if (!name) throw new Error("Usage: /gpu-cost-profile <profile>");
        const { file, data } = await costPolicy(ctx.cwd);
        if (!data.profiles?.[name]) throw new Error(`Unknown profile ${name}. Available: ${Object.keys(data.profiles ?? {}).sort().join(", ")}`);
        data.active_profile = name;
        await saveCostPolicy(file, data);
        await syncPrewalkOverlay(ctx.cwd, data);
        ctx.ui.notify(`GPU cost profile set to ${name}; model routing and the editing prewalk target are synchronized.`, "info");
      } catch (error) {
        ctx.ui.notify(error instanceof Error ? error.message : String(error), "error");
      }
    },
  });

  pi.registerCommand("gpu-prewalk", {
    description: "Override edit-subagent prewalk for a situation: /gpu-prewalk <situation> <auto|on|off>",
    handler: async (args, ctx) => {
      try {
        const [situation, mode] = words(args);
        if (!situation || !mode || !["auto", "on", "off"].includes(mode)) {
          throw new Error("Usage: /gpu-prewalk <situation> <auto|on|off>");
        }
        const { file, data } = await costPolicy(ctx.cwd);
        data.developer_overrides ??= {};
        data.developer_overrides.prewalk_situations ??= {};
        data.developer_overrides.prewalk_situations[situation] = mode;
        await saveCostPolicy(file, data);
        ctx.ui.notify(`Prewalk ${situation}=${mode}. This affects future implementation subagents only.`, "info");
      } catch (error) {
        ctx.ui.notify(error instanceof Error ? error.message : String(error), "error");
      }
    },
  });

  pi.registerCommand("spec-cost-route", {
    description: "Explain the cost/model/prewalk route for one approved SpecFlow task",
    handler: async (args, ctx) => {
      try {
        const [changeRaw, taskRaw] = words(args);
        const change = validId(changeRaw ?? "", "Change ID");
        const task = validId(taskRaw ?? "", "Task ID");
        await send(pi, [
          "Load and follow the specflow-cost-routing skill.",
          `Read the approved detailed task for change \`${change}\`, task \`${task}\`, and \`.omp/gpu-lab/cost-policy.json\`.`,
          "Resolve the exact role, agent, model fallback chain, thinking level, and prewalk decision without spawning or editing.",
          "Explain how the developer can override the decision in task frontmatter or with /gpu-prewalk and /gpu-cost-profile.",
        ].join("\n"));
      } catch (error) {
        ctx.ui.notify(error instanceof Error ? error.message : String(error), "error");
      }
    },
  });

  pi.registerCommand("spec-gpu-validate", {
    description: "Run the approved change's required GPU validation matrix",
    handler: async (args, ctx) => {
      try {
        const [changeRaw, taskRaw = "final"] = words(args);
        const change = validId(changeRaw ?? "", "Change ID");
        const task = validId(taskRaw, "Task ID");
        if (!(await exists(path.join(ctx.cwd, ".specs", change)))) throw new Error(`No .specs/${change} directory exists.`);
        await send(pi, [
          `Load and follow specflow-gpu-orchestrator and specflow-gpu-testing for change \`${change}\`, task \`${task}\`.`,
          "Read approved specification/task GPU target requirements and `.omp/gpu-lab` configuration.",
          "Use one shared run ID and a task batch of read-only CUDA/ROCm validators plus the portability reviewer.",
          "The coordinator is the sole source writer. Persist evidence under the change directory and do not waive failed required targets.",
        ].join("\n"));
      } catch (error) {
        ctx.ui.notify(error instanceof Error ? error.message : String(error), "error");
      }
    },
  });

  pi.registerCommand("spec-gpu-debug", {
    description: "Start a serial evidence-driven GPU debug escalation for a change",
    handler: async (args, ctx) => {
      try {
        const [changeRaw, hostRaw, program, runId] = words(args);
        const change = validId(changeRaw ?? "", "Change ID");
        const host = validId(hostRaw ?? "", "Host ID");
        if (!program) throw new Error("Usage: /spec-gpu-debug <change-id> <host-id> <program> [run-id]");
        await send(pi, [
          `Load and follow specflow-gpu-debugging for change \`${change}\` on host \`${host}\`.`,
          `Program: \`${program}\`${runId ? `; run ID: \`${validId(runId, "Run ID")}\`` : ""}.`,
          "First verify the failure was reproduced twice and call `gpu-labctl debug-plan`. Dispatch specflow-gpu-debugger only when preconditions hold.",
          "Use exactly one DAP root session. For unsupported CUDA device behavior, use the documented cuda-gdbserver fallback rather than guessing.",
        ].join("\n"));
      } catch (error) {
        ctx.ui.notify(error instanceof Error ? error.message : String(error), "error");
      }
    },
  });
}
