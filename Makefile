TEST_MODULES = \
	tests/test_workspace.py \
	tests/test_assistant_conversation.py \
	tests/test_assistant_threads.py \
	tests/test_dialogue_state.py \
	tests/test_fact_semantics.py \
	tests/test_provenance_fact_store.py \
	tests/test_medication_event_workflow.py \
	tests/test_model_runtime.py \
	tests/test_hybrid_retrieval.py \
	tests/test_decision_rag.py \
	tests/test_decision_rescue.py

.PHONY: test benchmark run check

test:
	@set -e; for test_file in $(TEST_MODULES); do \
		echo "==> $$test_file"; \
		pytest -q $$test_file; \
	done

benchmark:
	mkdir -p runs
	python scripts/evaluate_retrieval.py --json-output runs/fts5-baseline.json

run:
	bash scripts/start.sh

check:
	python scripts/check_public_release.py
