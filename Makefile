# OBSERVER — Make targets for common workflows.
#
# Run `make help` to see all targets. Most targets take CKPT= or DIR=:
#
#     make doctor                           # pre-flight check
#     make eval    CKPT=runs/x/model.pth    # single checkpoint
#     make sweep   DIR=runs/exp_001         # all .pth in DIR (non-recursive)
#     make latest  DIR=runs/                # newest .pth per subdir, recursive
#     make best    DIR=runs/                # sweep + auto-select top-3
#     make dry     DIR=runs/                # validate pipeline, no Isaac
#     make quick   DIR=runs/                # metrics only — no video, no tracking
#
# Tunables (override on command line):
#     WEIGHTS=hardware_safe|balanced|performance_first   (default: hardware_safe)
#     TOPK=N                                              (default: 3)

CKPT    ?=
DIR     ?=
WEIGHTS ?= hardware_safe
TOPK    ?= 3

.PHONY: help doctor eval sweep latest best dry quick

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?##/ { \
		printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

doctor: ## Pre-flight environment & config check
	observer doctor

eval: ## Evaluate a single checkpoint (CKPT=path)
	@[ -n "$(CKPT)" ] || { echo "Usage: make eval CKPT=path/to/model.pth"; exit 2; }
	observer --checkpoint $(CKPT)

sweep: ## Evaluate all checkpoints in DIR (non-recursive)
	@[ -n "$(DIR)" ] || { echo "Usage: make sweep DIR=runs/exp"; exit 2; }
	observer --checkpoint_dir $(DIR)

latest: ## Recursive sweep, newest checkpoint per subdirectory
	@[ -n "$(DIR)" ] || { echo "Usage: make latest DIR=runs/"; exit 2; }
	observer --checkpoint_dir $(DIR) --recursive --latest_only

best: ## Sweep + auto-select top-$(TOPK) (WEIGHTS=$(WEIGHTS))
	@[ -n "$(DIR)" ] || { echo "Usage: make best DIR=runs/"; exit 2; }
	observer --checkpoint_dir $(DIR) --recursive \
	    --auto_select --select_weights $(WEIGHTS) --deploy_top_k $(TOPK)

dry: ## Validate pipeline without launching Isaac
	@[ -n "$(DIR)" ] || { echo "Usage: make dry DIR=runs/"; exit 2; }
	observer --checkpoint_dir $(DIR) --dry_run

quick: ## Metrics only — skip video and experiment tracking
	@[ -n "$(DIR)" ] || { echo "Usage: make quick DIR=runs/"; exit 2; }
	observer --checkpoint_dir $(DIR) --skip_video --no_tracking
