# Part III — Operation contracts

Each compute operation owns exactly one normative section in this part. An
operation section fixes that operation's public ABI, logical layout, tensor and
index classification, backend capability matrix, workspace and control-status
protocol, admission, alias, and error rules, queued-data and deferred-failure
behavior, implementation recipe, and independent-reference obligations. It does
not restate Part II: where a shared contract already exists (device identity,
tensor ownership, views, queue and tokens, memory and workspace, error
categories, and conformance obligations), the operation section states only
what an implementer of that operation must additionally observe.

Operation sections are the single normative source for their operation. An
implementation, its declarations, its shared conformance cases, and the section
MUST stay synchronized; a defect found later is corrected in the section, the
affected implementation, and a behavioral regression together, followed by
revalidation of every completed backend port.
