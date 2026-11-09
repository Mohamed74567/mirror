.PHONY: help video demo smoke fast ablation test clean

help:
	@echo "Mirror pipeline targets:"
	@echo "  make video  - (re)generate the synthetic demo video + ground truth"
	@echo "  make demo   - run the full pipeline on the demo video"
	@echo "  make fast   - run the pipeline with a reduced optimisation budget"
	@echo "  make smoke  - very quick end-to-end sanity run"
	@echo "  make ablation - fused-loss ablation + robustness study (GT-scored)"
	@echo "  make test   - run the pytest suite"
	@echo "  make clean  - remove generated outputs and caches"

video:
	python scripts/make_demo_video.py

demo:
	python run_mirror.py --profile full

fast:
	python run_mirror.py --profile fast

smoke:
	python run_mirror.py --profile smoke

ablation:
	python scripts/ablation_study.py

test:
	python -m pytest

clean:
	rm -rf outputs models .pytest_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} \;
