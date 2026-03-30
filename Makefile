PYTHON = python3
SCRIPT = test_script.py
SPLIT_SCRIPT = src/split_data.py
TRAIN_CNN = src/train_cnn.py
TRAIN_MLP = src/train_mlp.py

MODES = \
	no_overlap_circle \
	no_overlap_circle_bw \
	no_overlap_triangle \
	no_overlap_triangle_bw \
	overlap_circle \
	overlap_circle_bw \
	overlap_triangle \
	overlap_triangle_bw

SPLIT_DIRS = \
	data/train/no_overlap_circle \
	data/train/no_overlap_circle_bw \
	data/train/no_overlap_triangle \
	data/train/no_overlap_triangle_bw \
	data/train/overlap_circle \
	data/train/overlap_circle_bw \
	data/train/overlap_triangle \
	data/train/overlap_triangle_bw

.PHONY: generate-data clear-data split-data train-cnn train-mlp train $(MODES)

generate-data: $(SPLIT_DIRS)

data/train/no_overlap_circle:
	@if [ ! -d data/train ]; then \
		echo "Generating dataset..."; \
		for mode in $(MODES); do \
			$(PYTHON) $(SCRIPT) $$mode; \
		done; \
		echo "Splitting dataset..."; \
		$(PYTHON) $(SPLIT_SCRIPT); \
	else \
		echo "Split dataset already exists; skipping generation."; \
	fi

$(filter-out data/train/no_overlap_circle,$(SPLIT_DIRS)): data/train/no_overlap_circle
	@:

split-data:
	$(PYTHON) $(SPLIT_SCRIPT)

clear-data:
	rm -rf data/no_overlap_circle
	rm -rf data/no_overlap_circle_bw
	rm -rf data/no_overlap_triangle
	rm -rf data/no_overlap_triangle_bw
	rm -rf data/overlap_circle
	rm -rf data/overlap_circle_bw
	rm -rf data/overlap_triangle
	rm -rf data/overlap_triangle_bw
	rm -rf data/train
	rm -rf data/val
	rm -rf data/test

train-cnn: generate-data
	$(PYTHON) $(TRAIN_CNN)

train-mlp: generate-data
	$(PYTHON) $(TRAIN_MLP)

train: train-cnn train-mlp

$(MODES):
	$(PYTHON) $(SCRIPT) $@
