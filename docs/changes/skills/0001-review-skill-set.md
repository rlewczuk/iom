Rework `cpp-inference-*` review skills in `.agents/skills` to introduce following changes:
* generate subtasks directly instead of generating single `review.md` file and then reprocess this again using `review-to-task` skill; see how `review-to-task` skill converts `review.md` to individual tasks, incorporate it into `cpp-inference-*` skill set and remove `review-to-task` skill
* make skills for individual specialists independently callable: developer should be able to either call full review (via orchestrator skill) or call just one specialist (for example, to find redundant and overengineered code)
* note that `review-to-task` contains a lot of cross checking, you may want to incorporate it into review skill set, possibly even have some post-process skill that will be called at the end of review (either full review or specialist review)
* put more emphasis to reducing complexity and preventing duplication and overengineering
