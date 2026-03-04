PYTHON = python3
SCRIPT = test_script.py

MODES = \
	no_overlap_circle \
	no_overlap_circle_bw \
	no_overlap_triangle \
	no_overlap_triangle_bw \
	overlap_circle \
	overlap_circle_bw \
	overlap_triangle \
	overlap_triangle_bw

.PHONY: all clean $(MODES)

all: $(MODES)

$(MODES):
	$(PYTHON) $(SCRIPT) $@

clean:
	rm -rf data/*