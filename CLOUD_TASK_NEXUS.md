# CLOUD TASK: ADAM COGNITIVE EVOLUTION (PHASE 2)
**Status:** CRITICAL
**Assigned to:** Nexus Engineer

## 1. OBJECTIVE
Prepare an optimized LoRA training environment for the local node "Adam" (Qwen 2.5 14b) on a Mac Mini M4 Pro (24GB RAM).

## 2. REQUIREMENTS
*   **Library:** [Apple MLX](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx-lm) (optimized for M4).
*   **Dataset:** `KNOWLEDGE_INCUBATOR/adam_golden_dataset_v1.jsonl` (Instruction-Following format).
*   **Target Model:** `qwen2.5-coder:14b` (Ollama compatible).
*   **Hardware Constraint:** 24GB Unified Memory.

## 3. EXPECTED ARTIFACTS
Please provide the following in the `eva_engine/forge/` directory:
1.  `mlx_lora_trainer.py`: A Python script to initiate training using MLX.
2.  `Modelfile_template`: A configuration file to merge the resulting LoRA adapter back into Ollama.
3.  `README_ADAM_TRAINING.md`: A 1-step instruction for Sergey to launch the process.

## 4. DESIGN FOCUS
*   **Integrity:** Ensure the script strictly follows the `[PYTHON_EXECUTE]` output standard.
*   **Performance:** Use 4-bit quantization during training if necessary to stay within 24GB.
*   **Validation:** Include a small evaluation loop to check accuracy against the golden dataset.

---
*Created by Eva Core via Gemini CLI*
*Timestamp: 2026-04-03*
