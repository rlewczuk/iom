import * as fs from "node:fs/promises";
import * as path from "node:path";
import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";

type StatusSnapshot = {
  changeId: string;
  specStatus: string;
  planStatus: string;
  stage: string;
  currentTask: string;
  blockedReason: string;
};

const CHANGE_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$/;

function now(): string {
  return new Date().toISOString();
}

function splitArgs(args: string): { changeId: string; rest: string } {
  const trimmed = args.trim();
  const firstSpace = trimmed.search(/\s/);
  if (firstSpace < 0) return { changeId: trimmed, rest: "" };
  return {
    changeId: trimmed.slice(0, firstSpace),
    rest: trimmed.slice(firstSpace).trim(),
  };
}

function assertChangeId(value: string): string {
  if (!value || !CHANGE_ID.test(value) || value === "." || value === "..") {
    throw new Error(
      "Change ID must be 1-80 characters using letters, numbers, '.', '_' or '-', " +
        "must start with a letter or number, and must not be '.' or '..'.",
    );
  }
  return value;
}

function stateDir(cwd: string, changeId: string): string {
  const specsRoot = path.resolve(cwd, ".specs");
  const target = path.resolve(specsRoot, changeId);
  if (target !== specsRoot && !target.startsWith(`${specsRoot}${path.sep}`)) {
    throw new Error("Resolved change path escaped the project .specs directory.");
  }
  return target;
}

async function exists(file: string): Promise<boolean> {
  try {
    await fs.access(file);
    return true;
  } catch {
    return false;
  }
}

async function readText(file: string): Promise<string> {
  try {
    return await fs.readFile(file, "utf8");
  } catch {
    return "";
  }
}

function field(markdown: string, key: string, fallback: string): string {
  const escaped = key.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = markdown.match(new RegExp(`^${escaped}:\\s*(.*?)\\s*$`, "m"));
  if (!match) return fallback;
  return match[1].replace(/^["']|["']$/g, "").trim() || fallback;
}

function specificationTemplate(changeId: string, initial: string): string {
  const request = initial || "_Developer: replace this placeholder with the initial specification._";
  return `---
change_id: ${changeId}
status: draft
revision: 1
updated_at: ${now()}
approved_at:
approval_evidence:
---

# ${changeId} — Specification

## Initial request

${request}

## Repository context

_To be populated by the refinement agent._

## Clarifications and decision log

_No decisions recorded yet._

## Final specification

_To be consolidated and updated after every material developer answer._

### Goals

### User-visible behavior

### Functional requirements

### Interfaces and data

### Error handling and edge cases

### Compatibility and migration

### Security, privacy, and operational constraints

### Observability

### Cost and model policy

_Record the preferred economy/balanced/quality profile, minimum model tiers for risky requirements, and whether edit-triggered prewalk is allowed. Developers retain per-task control through model_class, prewalk_policy, and prewalk_situation._

### Exploration and project knowledge policy

_Record whether parallel codebase exploration is permitted, whether external internet research is permitted, and which durable changes must update docs/agent-wiki/. Internet research defaults to disabled._

### GPU target matrix

_Record required CUDA/ROCm hosts, architectures, debug/release profiles, benchmark requirements, and acceptable environment exceptions. Use 'none' when GPU validation is not applicable._

### Remote debugging policy

_Record which failures require debugger escalation and any source-path constraints._

## Acceptance criteria

## Non-goals

## Open questions

_To be populated by the refinement agent._

## Approval history

_Not approved._
`;
}

function progressTemplate(changeId: string): string {
  return `---
change_id: ${changeId}
stage: specification
spec_status: draft
plan_status: absent
current_task:
blocked_reason:
updated_at: ${now()}
---

# ${changeId} — Progress

## Approval gates

- Specification approval: pending
- Plan approval: pending

## Current position

Specification refinement has not completed.

## Completed tasks

_None._

## Pending tasks

_Not planned._

## Durable discoveries

_None._

## Blockers and escalations

_None._

## Resume instructions

Resume specification refinement from \`.specs/${changeId}/specification.md\`.
`;
}

async function ensureChange(cwd: string, changeId: string, initial: string): Promise<{
  dir: string;
  createdSpec: boolean;
  createdProgress: boolean;
}> {
  const dir = stateDir(cwd, changeId);
  await fs.mkdir(path.join(dir, "tasks"), { recursive: true });
  await fs.mkdir(path.join(dir, "reviews"), { recursive: true });
  await fs.mkdir(path.join(dir, "reports"), { recursive: true });
  await fs.mkdir(path.join(dir, "gpu-lab"), { recursive: true });

  const spec = path.join(dir, "specification.md");
  const progress = path.join(dir, "progress.md");
  let createdSpec = false;
  let createdProgress = false;

  if (!(await exists(spec))) {
    await fs.writeFile(spec, specificationTemplate(changeId, initial), {
      encoding: "utf8",
      flag: "wx",
    });
    createdSpec = true;
  }

  if (!(await exists(progress))) {
    await fs.writeFile(progress, progressTemplate(changeId), {
      encoding: "utf8",
      flag: "wx",
    });
    createdProgress = true;
  }

  return { dir, createdSpec, createdProgress };
}

async function snapshot(cwd: string, changeId: string): Promise<StatusSnapshot> {
  const dir = stateDir(cwd, changeId);
  const spec = await readText(path.join(dir, "specification.md"));
  const tasks = await readText(path.join(dir, "tasks.md"));
  const progress = await readText(path.join(dir, "progress.md"));

  return {
    changeId,
    specStatus: field(spec, "status", spec ? "unknown" : "missing"),
    planStatus: field(tasks, "status", tasks ? "unknown" : "absent"),
    stage: field(progress, "stage", progress ? "unknown" : "missing"),
    currentTask: field(progress, "current_task", "none"),
    blockedReason: field(progress, "blocked_reason", "none"),
  };
}

async function listChanges(cwd: string): Promise<string[]> {
  const root = path.resolve(cwd, ".specs");
  try {
    const entries = await fs.readdir(root, { withFileTypes: true });
    return entries
      .filter((entry) => entry.isDirectory() && CHANGE_ID.test(entry.name))
      .map((entry) => entry.name)
      .sort();
  } catch {
    return [];
  }
}

function statusText(value: StatusSnapshot): string {
  return [
    `SpecFlow: ${value.changeId}`,
    `stage=${value.stage}`,
    `spec=${value.specStatus}`,
    `plan=${value.planStatus}`,
    `current_task=${value.currentTask}`,
    `blocked=${value.blockedReason}`,
  ].join(" | ");
}

async function prompt(pi: ExtensionAPI, text: string): Promise<void> {
  await pi.sendUserMessage(text);
}

export default function specflow(pi: ExtensionAPI) {
  pi.setLabel("Superpowers SpecFlow");

  pi.registerCommand("spec-start", {
    description: "Initialize and refine .specs/<change-id>/specification.md",
    handler: async (args, ctx) => {
      try {
        const parsed = splitArgs(args);
        const changeId = assertChangeId(parsed.changeId);
        const result = await ensureChange(ctx.cwd, changeId, parsed.rest);
        ctx.ui.notify(
          `${changeId}: ${result.createdSpec ? "created specification" : "using existing specification"}`,
          "info",
        );
        await prompt(
          pi,
          [
            `Load and follow the specflow-refinement skill for change \`${changeId}\`.`,
            `Project root: \`${ctx.cwd}\`.`,
            `Durable state directory: \`.specs/${changeId}\`.`,
            "Inspect repository context first. Ask exactly one ambiguity-resolving question at a time.",
            "Update specification.md after every material answer.",
            "Do not plan or implement before explicit written-specification approval.",
          ].join("\n"),
        );
      } catch (error) {
        ctx.ui.notify(error instanceof Error ? error.message : String(error), "error");
      }
    },
  });

  pi.registerCommand("spec-plan", {
    description: "Create or revise tasks.md and detailed task files for an approved specification",
    handler: async (args, ctx) => {
      try {
        const changeId = assertChangeId(splitArgs(args).changeId);
        const value = await snapshot(ctx.cwd, changeId);
        if (value.specStatus !== "approved") {
          throw new Error(
            `${changeId}: specification status is '${value.specStatus}', not 'approved'. ` +
              "Resume specification refinement first.",
          );
        }
        await prompt(
          pi,
          [
            `Load and follow the specflow-planning skill for change \`${changeId}\`.`,
            `Use \`.specs/${changeId}/specification.md\`, \`tasks.md\`, \`tasks/\`, and \`progress.md\`.`,
            "Do not edit production code.",
            "Create or correct the task index and one self-contained detailed Markdown file per task.",
            "Stop at the explicit plan-review gate unless the developer explicitly approves and asks to proceed.",
          ].join("\n"),
        );
      } catch (error) {
        ctx.ui.notify(error instanceof Error ? error.message : String(error), "error");
      }
    },
  });

  pi.registerCommand("spec-implement", {
    description: "Execute an approved SpecFlow plan with implementation, review, and test subagents",
    handler: async (args, ctx) => {
      try {
        const changeId = assertChangeId(splitArgs(args).changeId);
        const value = await snapshot(ctx.cwd, changeId);
        if (value.specStatus !== "approved" || value.planStatus !== "approved") {
          throw new Error(
            `${changeId}: implementation requires spec=approved and plan=approved; ` +
              `current spec=${value.specStatus}, plan=${value.planStatus}.`,
          );
        }
        await prompt(
          pi,
          [
            `Load and follow the specflow-execution skill for change \`${changeId}\`.`,
            `Resume only from durable files under \`.specs/${changeId}\`.`,
            "Dispatch fresh specflow-implementer subagents, followed by independent specflow-reviewer and specflow-tester subagents.",
            "Run final whole-change review and testing.",
            "Do not generate a final summary report.",
          ].join("\n"),
        );
      } catch (error) {
        ctx.ui.notify(error instanceof Error ? error.message : String(error), "error");
      }
    },
  });

  pi.registerCommand("spec-status", {
    description: "Show durable SpecFlow status or list known changes",
    handler: async (args, ctx) => {
      try {
        const parsed = splitArgs(args);
        if (!parsed.changeId) {
          const changes = await listChanges(ctx.cwd);
          ctx.ui.notify(
            changes.length ? `SpecFlow changes: ${changes.join(", ")}` : "No .specs change directories found.",
            "info",
          );
          return;
        }
        const changeId = assertChangeId(parsed.changeId);
        const value = await snapshot(ctx.cwd, changeId);
        ctx.ui.notify(statusText(value), value.stage === "blocked" ? "warning" : "info");
      } catch (error) {
        ctx.ui.notify(error instanceof Error ? error.message : String(error), "error");
      }
    },
  });

  pi.registerCommand("spec-resume", {
    description: "Resume a change from specification, planning, implementation, or blocked state",
    handler: async (args, ctx) => {
      try {
        const changeId = assertChangeId(splitArgs(args).changeId);
        const dir = stateDir(ctx.cwd, changeId);
        if (!(await exists(dir))) {
          throw new Error(`${changeId}: no state directory exists. Run /spec-start first.`);
        }
        const value = await snapshot(ctx.cwd, changeId);
        ctx.ui.notify(statusText(value), "info");
        await prompt(
          pi,
          [
            `Load and follow the specflow-orchestrator skill for change \`${changeId}\`.`,
            `Read \`.specs/${changeId}/progress.md\`, specification and plan frontmatter, then only the next relevant task and evidence files.`,
            "Reconstruct state from files rather than conversation memory.",
            `Observed status: stage=${value.stage}, spec=${value.specStatus}, plan=${value.planStatus}, current_task=${value.currentTask}.`,
            "Route to refinement, planning, execution, or blocker resolution according to the durable state.",
          ].join("\n"),
        );
      } catch (error) {
        ctx.ui.notify(error instanceof Error ? error.message : String(error), "error");
      }
    },
  });
}
