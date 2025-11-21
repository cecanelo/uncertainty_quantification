.PHONY: test train eval submit

# Partition presets
TEST_FLAGS=-p TEST -t 00:10:00
STUD_FLAGS=-p STUD -t 00:30:00 -c 4 --mem=8G

# Most-recent OUTDIR helper (hyphen pattern)
LATEST_OUTDIR=$(shell ls -dt outputs/*-* 2>/dev/null | head -n 1)

# Ensure logs dir exists
LOGS_MK=mkdir -p logs

# ---------------- Legacy / convenience ----------------

test:
	$(LOGS_MK)
	RUN_MODE=debug CONFIG_PATH=configs/default.yaml \
	sbatch $(TEST_FLAGS) \
	  --output="logs/%x-debug-%j.out" \
	  --error="logs/%x-debug-%j.err" \
	  -J practice \
	  scripts/submit_gpu.sbatch

train:
	$(LOGS_MK)
	RUN_MODE=train CONFIG_PATH=configs/mnist_train.yaml \
	sbatch $(STUD_FLAGS) \
	  --output="logs/%x-train-%j.out" \
	  --error="logs/%x-train-%j.err" \
	  -J practice \
	  scripts/submit_gpu.sbatch

# EVAL RULE: reuse training OUTDIR, and force eval logs to use the TRAIN jobid
# Usage: OUTDIR=outputs/practice-debug-271097 make eval
eval:
ifndef OUTDIR
	$(error Please set OUTDIR to the training folder, e.g. OUTDIR=outputs/practice-debug-271097)
endif
	$(LOGS_MK)
	@b=$$(basename "$(OUTDIR)"); \
	name=$$(echo $$b | awk -F- '{print $$1}'); \
	mode=$$(echo $$b | awk -F- '{print $$2}'); \
	tjid=$$(echo $$b | awk -F- '{print $$NF}'); \
	if [ -z "$$name" ] || [ -z "$$tjid" ]; then \
	  echo "[error] Could not parse OUTDIR='$(OUTDIR)'. Expected outputs/<job-name>-<mode>-<jobid>." >&2; \
	  exit 2; \
	fi; \
	echo "[eval] OUTDIR=$(OUTDIR)  name=$$name  train_mode=$$mode  train_jobid=$$tjid"; \
	RUN_MODE=eval CONFIG_PATH=configs/mnist_train.yaml OUTDIR="$(OUTDIR)" \
	LOG_BASENAME="$$name-eval-$$tjid" \
	sbatch $(STUD_FLAGS) \
	  --output="logs/$$name-eval-$$tjid.out" \
	  --error="logs/$$name-eval-$$tjid.err" \
	  -J "$$name" \
	  scripts/submit_gpu.sbatch --make_report --panels 1,2,3,4,5

# ---------------- Single entry point ----------------

# Usage examples:
#   make submit MODE=debug
#   make submit MODE=train CFG=configs/mnist_train.yaml
#   OUTDIR=outputs/practice-debug-271097 make submit MODE=eval ARGS="--make_report"
MODE ?= debug
CFG  ?= configs/default.yaml
ARGS ?=

ifeq ($(MODE),debug)
  PART=$(TEST_FLAGS)
  MFLAG=debug
else ifeq ($(MODE),train)
  PART=$(STUD_FLAGS)
  MFLAG=train
else ifeq ($(MODE),eval)
  PART=$(STUD_FLAGS)
  MFLAG=eval
else
  $(error Unknown MODE '$(MODE)'; use debug|train|eval)
endif

submit:
	$(LOGS_MK)
ifeq ($(MODE),eval)
ifndef OUTDIR
	$(error For MODE=eval, set OUTDIR to the training folder, e.g. OUTDIR=outputs/practice-debug-271097)
endif
	@b=$$(basename "$(OUTDIR)"); \
	name=$$(echo $$b | awk -F- '{print $$1}'); \
	tjid=$$(echo $$b | awk -F- '{print $$NF}'); \
	if [ -z "$$name" ] || [ -z "$$tjid" ]; then \
	  echo "[error] Could not parse OUTDIR='$(OUTDIR)'. Expected outputs/<job-name>-<mode>-<jobid>." >&2; \
	  exit 2; \
	fi; \
	RUN_MODE=$(MODE) CONFIG_PATH=$(CFG) OUTDIR="$(OUTDIR)" \
	LOG_BASENAME="$$name-$(MFLAG)-$$tjid" \
	sbatch $(PART) \
	  --output="logs/$$name-$(MFLAG)-$$tjid.out" \
	  --error="logs/$$name-$(MFLAG)-$$tjid.err" \
	  -J "$$name" \
	  scripts/submit_gpu.sbatch $(ARGS)
else
	RUN_MODE=$(MODE) CONFIG_PATH=$(CFG) \
	sbatch $(PART) \
	  --output="logs/%x-$(MFLAG)-%j.out" \
	  --error="logs/%x-$(MFLAG)-%j.err" \
	  -J practice \
	  scripts/submit_gpu.sbatch $(ARGS)
endif

.PHONY: hpo hpo-dry hpo-status hpo-tail hpo-export

# HPO convenience vars
HPO_CFG      ?= configs/hpo_config.yaml
HPO_JOBID    ?=
HPO_JOB_ROOT ?=

# Submit tuning (non-blocking: sbatch returns immediately)
hpo:
	@mkdir -p logs optuna_studies outputs/optuna
	python scripts/hpo_runner.py --config $(HPO_CFG)

# Preview the sbatch plan and job root without submitting
hpo-dry:
	@mkdir -p logs optuna_studies outputs/optuna
	python scripts/hpo_runner.py --config $(HPO_CFG) --dry-run

# Show scheduler status for a given JOBID
hpo-status:
ifndef HPO_JOBID
	$(error Set HPO_JOBID, e.g. make hpo-status HPO_JOBID=276782)
endif
	squeue -j $(HPO_JOBID) -o "%10i %8T %10M %5D %R %j"

# Tail logs live for a given JOBID
hpo-tail:
ifndef HPO_JOBID
	$(error Set HPO_JOBID, e.g. make hpo-tail HPO_JOBID=276782)
endif
	tail -F logs/*_$(HPO_JOBID)_*.out

# Export best hyperparameters (either by JOBID or by explicit job root)
hpo-export:
	@set -e; \
	if [ -n "$(HPO_JOBID)" ]; then \
	  python scripts/hpo_export_best.py --config $(HPO_CFG) --job-id $(HPO_JOBID); \
	elif [ -n "$(HPO_JOB_ROOT)" ]; then \
	  python scripts/hpo_export_best.py --config $(HPO_CFG) --job-root "$(HPO_JOB_ROOT)"; \
	else \
	  echo "Set HPO_JOBID=<id> or HPO_JOB_ROOT=outputs/optuna/<job_tag>"; exit 2; \
	fi

.PHONY: hpo-analyze

# Analyze an HPO job or a specific job root.
# Usage:
#   make hpo-analyze HPO_JOBID=276812
#   make hpo-analyze HPO_JOB_ROOT="outputs/optuna/<job_tag>_NA"
HPO_ANALYZE_CFG ?= configs/hpo_config.yaml
HPO_JOBID       ?=
HPO_JOB_ROOT    ?=

hpo-analyze:
	@if [ -n "$(HPO_JOBID)" ]; then \
		echo "[analyze] by JOBID=$(HPO_JOBID)"; \
		python scripts/analyze_hpo_job.py --job-id $(HPO_JOBID); \
	elif [ -n "$(HPO_JOB_ROOT)" ]; then \
		echo "[analyze] by JOB_ROOT=$(HPO_JOB_ROOT)"; \
		python scripts/analyze_hpo_job.py --job-root "$(HPO_JOB_ROOT)"; \
	else \
		echo "Set HPO_JOBID=<id> or HPO_JOB_ROOT=outputs/optuna/<job_tag>_NA"; exit 2; \
	fi

.PHONY: train-best-local train-best

# Optional overrides:
#   BEST_JSON : explicit path to best_hparams.json
#               (default: uses outputs/optuna/<study>/best/best_hparams.json from the HPO YAML)
#   HPO_CFG   : which HPO YAML to read study/storage from
HPO_CFG   ?= configs/hpo_config.yaml
BEST_JSON ?=

# Run training locally (blocks)
train-best-local:
	@mkdir -p logs
ifneq ($(strip $(BEST_JSON)),)
	python scripts/hpo_train_best.py --config $(HPO_CFG) --best-json "$(BEST_JSON)"
else
	python scripts/hpo_train_best.py --config $(HPO_CFG)
endif

# Submit training to SLURM (non-blocking). Adjust resources if needed.
# Usage examples:
#   make train-best
#   make train-best BEST_JSON=outputs/optuna/<job_tag>/best/best_hparams.json
train-best:
	@mkdir -p logs
ifneq ($(strip $(BEST_JSON)),)
	sbatch -p STUD -t 00:10:00 -c 2 --mem=4G --gres=gpu:1 \
	  -J train-best \
	  -o logs/train-best_%j.out -e logs/train-best_%j.err \
	  --wrap 'source $$HOME/miniconda3/etc/profile.d/conda.sh; conda activate thesis; cd $$HOME/uq_demo; python scripts/hpo_train_best.py --config $(HPO_CFG) --best-json "$(BEST_JSON)"'
else
	sbatch -p STUD -t 00:10:00 -c 2 --mem=4G --gres=gpu:1 \
	  -J train-best \
	  -o logs/train-best_%j.out -e logs/train-best_%j.err \
	  --wrap 'source $$HOME/miniconda3/etc/profile.d/conda.sh; conda activate thesis; cd $$HOME/uq_demo; python scripts/hpo_train_best.py --config $(HPO_CFG)'
endif
