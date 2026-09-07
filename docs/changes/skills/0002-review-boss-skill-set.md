There is a set of skills `.agents/skills/cpp-inference-*` that are reviewing codebase. 
Those skills now require strong agent that consumes lot of tokens. 
Please adapt those review skills to use `boss` skill implemented in `.omp/skills/boss` in
order to redirect as much of LLM traffic from expensive frontier model to cheper flash models. After that, review process should offload as much simpler tasks (eg. exploration)
to models labeled `@task` or `@smol`, use `@slow` to orchestrate process and `@advisor` for deep thinking and key decisions. Model labeled `@advisor` should perform only deep thinking, should not access source files directly. Model `@slow` should also conserve tokens and use `@smol` or `@task` to perform searches across codebase, should minimize direct source code accesses to cases where it knows exactly what files and what parts of them to read (thus it is cheaper than calling subagents to do it). 

You are free to also modify boss skill and its defined agents in `.omp/skills` and `.omp/agents` if you think it has to be improved to work with code review skills.