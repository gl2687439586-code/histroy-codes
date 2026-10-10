# Third-party method notice

This experiment reimplements the Adaptive-k selection rule from:

> Chihiro Taguchi, Seiji Maekawa, and Nikita Bhutani. Efficient Context
> Selection for Long-Context QA: No Tuning, No Iteration, Just Adaptive-k.
> EMNLP 2025 Main. https://aclanthology.org/2025.emnlp-main.1017/

Official implementation:
https://github.com/megagonlabs/adaptive-k-retrieval (BSD-3-Clause).

The implementation in `adaptive_k.py` mirrors the official `largest_gap`,
`ignore_extreme_tail=0.1`, and `retrieve_more=5` behavior.  The original QA
template is reproduced in `qa_template.txt` for protocol transparency.
