.PHONY: clean_overlap clean_overlap_bw clean_no_overlap clean_no_overlap_bw

clean_overlap:
	rm -rf data/images_overlap_triangle

clean_overlap_bw:
	rm -rf data/images_overlap_triangle_bw

clean_no_overlap:
	rm -rf data/images_no_overlap

clean_no_overlap_bw:
	rm -rf data/images_no_overlap_bw

overlap_triangle:
	python3 test_script.py overlap_triangle

overlap_triangle_bw:
	python3 test_script.py overlap_triangle_bw

overlap_circle:
	python3 test_script.py overlap_circle

overlap_circle_bw:
	python3 test_script.py overlap_circle_bw

no_overlap:
	python3 test_script.py no_overlap

no_overlap_bw:
	python3 test_script.py no_overlap_bw

train_mlp:
	python3 src/train_mlp.py

train_cnn:
	python3 src/train_cnn.py

confusion_matrix_mlp:
	python3 src/generate_confusion_matrix.py

confusion_matrix_cnn:
	python3 src/confusion_matrix_cnn.py