import * as fs from "node:fs/promises";
import * as path from "node:path";
import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";

type SessionState = {
  explorersEnabled: boolean;
  internetEnabled: boolean;
  oneShotLocal: number;
  oneShotWeb: number;
};
type Policy = {
  version?: number;
  defaults?: {
    explorers_enabled?: boolean;
    internet_enabled?: boolean;
    max_parallel_explorers?: number;
    docs_root?: string;
    docs_update?: string;
  };
};

const ENTRY = "specflow-exploration-state";
let state: SessionState = { explorersEnabled: true, internetEnabled: false, oneShotLocal: 0, oneShotWeb: 0 };
let initialized = false;
let policyActive = false;
let maxParallelExplorers = 4;

function parseToggle(args: string): "on" | "off" | "status" {
  const value = args.trim().toLowerCase() || "status";
  if (value !== "on" && value !== "off" && value !== "status") throw new Error("Expected on, off, or status.");
  return value;
}

async function readPolicy(cwd: string): Promise<{ data: Policy; active: boolean }> {
  const file = path.join(cwd, ".omp", "gpu-lab", "exploration-policy.json");
  try {
    const data = JSON.parse(await fs.readFile(file, "utf8")) as Policy;
    if (data.version !== 1 || !data.defaults) throw new Error(`${file} is not a version-1 exploration policy.`);
    return { data, active: true };
  } catch (error: any) {
    if (error?.code === "ENOENT") {
      return { data: { version: 1, defaults: { explorers_enabled: true, internet_enabled: false, docs_root: "docs/agent-wiki" } }, active: false };
    }
    throw error;
  }
}

async function restore(ctx: any): Promise<void> {
  const loaded = await readPolicy(ctx.cwd);
  const policy = loaded.data;
  policyActive = loaded.active;
  maxParallelExplorers = Math.max(1, Math.min(16, policy.defaults?.max_parallel_explorers ?? 4));
  state = {
    explorersEnabled: policy.defaults?.explorers_enabled !== false,
    internetEnabled: policy.defaults?.internet_enabled === true,
    oneShotLocal: 0,
    oneShotWeb: 0,
  };
  for (const entry of ctx.sessionManager.getBranch()) {
    if (entry.type === "custom" && entry.customType === ENTRY && entry.data) {
      state = { ...state, ...(entry.data as Partial<SessionState>) };
    }
  }
  initialized = true;
}

function persist(pi: ExtensionAPI): void {
  pi.appendEntry(ENTRY, { ...state });
}

function taskAgents(input: unknown): string[] {
  if (!input || typeof input !== "object") return [];
  const value = input as { agent?: unknown; tasks?: unknown };
  const out: string[] = [];
  if (typeof value.agent === "string") out.push(value.agent);
  if (Array.isArray(value.tasks)) {
    for (const item of value.tasks) {
      if (item && typeof item === "object" && typeof (item as { agent?: unknown }).agent === "string") {
        out.push((item as { agent: string }).agent);
      }
    }
  }
  return out;
}

function statusText(): string {
  return `explorers=${state.explorersEnabled ? "on" : "off"} | internet=${state.internetEnabled ? "on" : "off"} | max-parallel=${maxParallelExplorers} | one-shot-local=${state.oneShotLocal} | one-shot-web=${state.oneShotWeb}`;
}

export default function exploration(pi: ExtensionAPI) {
  const restoreEvent = async (_event: unknown, ctx: any) => {
    try {
      // Task/subagent sessions run in-process. Do not reset a parent session's
      // runtime gate to project defaults when a headless child starts.
      if (!ctx.hasUI && initialized) return;
      await restore(ctx);
    } catch (error) { ctx.ui.notify(error instanceof Error ? error.message : String(error), "error"); }
  };
  pi.on("session_start", restoreEvent);
  pi.on("session_branch", restoreEvent);
  pi.on("session_tree", restoreEvent);
  pi.on("session_switch", restoreEvent);

  pi.on("tool_call", async (event: any) => {
    if (event.toolName !== "task" || !policyActive) return;
    const agents = taskAgents(event.input);
    const localCount = agents.filter((name) => name === "scout").length;
    const webCount = agents.filter((name) => name === "librarian").length;
    if (!localCount && !webCount) return;
    if (localCount + webCount > maxParallelExplorers) {
      return { block: true, reason: `Exploration batch requests ${localCount + webCount} agents; policy maximum is ${maxParallelExplorers}.` };
    }

    if (!state.explorersEnabled && (localCount > state.oneShotLocal || webCount > state.oneShotWeb)) {
      return { block: true, reason: "Exploration agents are disabled for this session. Use /explorers on or /explore-once <question>." };
    }
    if (webCount > 0 && !state.internetEnabled && webCount > state.oneShotWeb) {
      return { block: true, reason: "Internet research is disabled for this session. Use /explorer-web on or /explore-once --web <question>." };
    }

    let remainingLocal = state.oneShotLocal;
    let remainingWeb = state.oneShotWeb;
    for (const name of agents) {
      if (name === "librarian" && remainingWeb > 0) remainingWeb -= 1;
      else if (name === "scout" && remainingLocal > 0) remainingLocal -= 1;
    }
    if (remainingLocal !== state.oneShotLocal || remainingWeb !== state.oneShotWeb) {
      state.oneShotLocal = remainingLocal;
      state.oneShotWeb = remainingWeb;
      persist(pi);
    }
  });

  pi.registerCommand("explorers", {
    description: "Enable, disable, or show exploration-agent availability for the current OMP session",
    handler: async (args, ctx) => {
      try {
        if (!policyActive) throw new Error("Exploration policy is not initialized in this project. Run /gpu-lab-init.");
        const value = parseToggle(args);
        if (value !== "status") {
          state.explorersEnabled = value === "on";
          persist(pi);
        }
        ctx.ui.notify(statusText(), "info");
      } catch (error) { ctx.ui.notify(error instanceof Error ? error.message : String(error), "error"); }
    },
  });

  pi.registerCommand("explorer-web", {
    description: "Enable, disable, or show external internet research for the current OMP session",
    handler: async (args, ctx) => {
      try {
        if (!policyActive) throw new Error("Exploration policy is not initialized in this project. Run /gpu-lab-init.");
        const value = parseToggle(args);
        if (value !== "status") {
          state.internetEnabled = value === "on";
          persist(pi);
        }
        ctx.ui.notify(statusText(), "info");
      } catch (error) { ctx.ui.notify(error instanceof Error ? error.message : String(error), "error"); }
    },
  });

  pi.registerCommand("explore-once", {
    description: "Run one focused local scout, or one native web librarian with --web, without changing session defaults",
    handler: async (args, ctx) => {
      if (!policyActive) { ctx.ui.notify("Exploration policy is not initialized in this project. Run /gpu-lab-init.", "error"); return; }
      const tokens = args.trim().split(/\s+/).filter(Boolean);
      const web = tokens[0] === "--web";
      if (web) tokens.shift();
      const question = tokens.join(" ").trim();
      if (!question) { ctx.ui.notify("Usage: /explore-once [--web] <focused question>", "error"); return; }
      if (web) state.oneShotWeb += 1; else state.oneShotLocal += 1;
      persist(pi);
      const role = web ? "external_librarian" : "explorer";
      const agent = web ? "librarian" : "scout";
      await pi.sendUserMessage([
        `Run exactly one one-shot exploration task using agent \`${agent}\`.`,
        `Resolve role \`${role}\` through \`.omp/gpu-lab/bin/gpu-lab-cost recommend --role ${role} --model-class smol --risk low --situation read-only --prewalk off\` and pass the returned model explicitly.`,
        `Question: ${question}`,
        "Return the exploration result directly. Do not edit files or start the ordinary SpecFlow execution workflow.",
      ].join("\n"));
    },
  });

  pi.registerCommand("docs-wiki-status", {
    description: "Show project agent-wiki paths and current exploration defaults/session gates",
    handler: async (_args, ctx) => {
      try {
        const loaded = await readPolicy(ctx.cwd);
        if (!loaded.active) throw new Error("Exploration policy is not initialized in this project. Run /gpu-lab-init.");
        const docsRoot = loaded.data.defaults?.docs_root ?? "docs/agent-wiki";
        const root = path.join(ctx.cwd, docsRoot);
        const index = await fs.stat(path.join(root, "index.md")).then(() => true).catch(() => false);
        const log = await fs.stat(path.join(root, "log.md")).then(() => true).catch(() => false);
        ctx.ui.notify(`${statusText()} | docs=${docsRoot} index=${index} log=${log}`, "info");
      } catch (error) { ctx.ui.notify(error instanceof Error ? error.message : String(error), "error"); }
    },
  });

  pi.registerCommand("docs-wiki-update", {
    description: "Ask the docs librarian to synchronize durable project knowledge for a change/task",
    handler: async (args, ctx) => {
      if (!policyActive) { ctx.ui.notify("Exploration policy is not initialized in this project. Run /gpu-lab-init.", "error"); return; }
      const target = args.trim();
      if (!target) { ctx.ui.notify("Usage: /docs-wiki-update <change-id> [task-id]", "error"); return; }
      await pi.sendUserMessage([
        "Load and follow the specflow-doc-librarian skill.",
        `Synchronize the project agent wiki for: ${target}.`,
        "Resolve role `doc_librarian` through gpu-lab-cost and spawn `specflow-doc-librarian` with the explicit model.",
        "Run it serially. Review the resulting docs-only diff before claiming completion.",
      ].join("\n"));
    },
  });
}
