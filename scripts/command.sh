#!/bin/bash
python build_sym_alignment.py --fast_align_dir ~/download/fast_align/build/ --mosesdecoder_dir ~/download/mo --source_file ../data/qqData/train.x --target_file ../data/qqData/train.y --output_dir dict
