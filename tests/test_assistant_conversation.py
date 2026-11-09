"""Intent routing, conversational follow-up, and answer-quality regressions."""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.main import create_app

VAULT_PASS = "conversation-vault-passphrase"
USER_PASS = "conversation-admin-password"


class BaseModelDouble:
    base_url = "http://127.0.0.1:10080/v1"
    model = "assistant-conversation-test-double"
    remote = False

    def __init__(self):
        self.calls: list[dict] = []

    def status(self):
        return {
            "available": True,
            "models": [self.model],
            "configured_model": self.model,
            "endpoint": self.base_url,
            "remote": False,
        }

    def summarize(self, text):
        return "Draft summary."

    def transcribe(self, image):
        return "Draft transcription."

    def answer(self, question, context):
        self.calls.append({"question": question, "context": context, "history": []})
        return "The chart contains reviewed information. [N1]"

    def close(self):
        pass


class EchoOnlyModel(BaseModelDouble):
    def answer_with_history(self, question, context, history):
        self.calls.append({"question": question, "context": context, "history": history})
        return "[R1] outside-records-synthetic.txt (version 1, excerpt)."


class NaturalConversationModel(BaseModelDouble):
    def answer_with_history(self, question, context, history):
        self.calls.append({"question": question, "context": context, "history": history})
        if history:
            return "The chart does not document when the report was originally requested. [R1]"
        return "The reviewed findings are consistent with the documented chart summary. [N1]"


class UnsafeRecommendationModel(BaseModelDouble):
    def answer_with_history(self, question, context, history):
        self.calls.append({"question": question, "context": context, "history": history})
        return "The patient should take paracetamol for a fever."


class RawDumpModel(BaseModelDouble):
    def answer_with_history(self, question, context, history):
        self.calls.append({"question": question, "context": context, "history": history})
        block = (
            "DEMO DATA — FICTIONAL PATIENT\nPatient: Morgan Avery\nDOB: 1988-04-12\n"
            "MRN: SYN-1001\nDocument type: outside records review\n"
            "Lisinopril 10 mg daily and cetirizine 10 mg as needed are listed."
        )
        return "\n".join([block, block, block])


def open_workspace(tmp_path, model: BaseModelDouble) -> tuple[TestClient, dict]:
    app = create_app(tmp_path, model)
    client = TestClient(app)
    client.__enter__()
    assert client.post('/api/setup', json={
        "username": "admin", "password": USER_PASS, "vault_password": VAULT_PASS,
    }).status_code == 200
    login = client.post('/api/login', json={"username": "admin", "password": USER_PASS})
    assert login.status_code == 200
    client.headers['Authorization'] = 'Bearer ' + login.json()['token']
    assert client.post('/api/seed').status_code == 201
    patient = next(row for row in client.get('/api/patients').json() if row['mrn'] == 'SYN-1001')
    return client, patient


def current_revision(client: TestClient, patient_id: str) -> int:
    return client.get(f'/api/patients/{patient_id}').json()['patient']['revision']


def ask(client: TestClient, patient_id: str, question: str, history=None):
    return client.post(f'/api/patients/{patient_id}/ask', json={
        "question": question,
        "expected_revision": current_revision(client, patient_id),
        "history": history or [],
    })


def test_common_overview_uses_structured_chart_answer_not_tiny_model(tmp_path):
    model = EchoOnlyModel(); client, patient = open_workspace(tmp_path, model)
    try:
        response = ask(client, patient['id'], 'What information is documented?')
        assert response.status_code == 200, response.text
        data = response.json()
        assert data['answer_origin'] == 'provenance_fact_store'
        assert data['resolved_intent'] == 'chart_overview'
        assert data['model_used'] is False
        assert model.calls == []
        answer = data['answer'].casefold()
        assert 'hypertension' in answer and 'lisinopril' in answer
        assert 'cardiology report' in answer and 'outstanding' in answer
        assert all(term not in answer for term in ['version 1', 'excerpt', '[r1]', 'synthetic', 'fictional', 'mrn'])
    finally:
        client.__exit__(None, None, None)


def test_pending_report_question_returns_specific_answer(tmp_path):
    model = NaturalConversationModel(); client, patient = open_workspace(tmp_path, model)
    try:
        response = ask(client, patient['id'], 'Which outside report is still missing?')
        assert response.status_code == 200
        data = response.json()
        assert data['answer_origin'] == 'provenance_fact_store'
        assert data['resolved_intent'] == 'pending_items'
        assert data['model_used'] is False and model.calls == []
        answer = data['answer'].casefold()
        assert 'outside cardiology report' in answer and 'still pending' in answer
        assert 'no interval changes' not in answer
        assert 1 <= len(data['sources']) <= 2
    finally:
        client.__exit__(None, None, None)


def test_pending_confirmation_follow_up_understands_context(tmp_path):
    model = NaturalConversationModel(); client, patient = open_workspace(tmp_path, model)
    try:
        first = ask(client, patient['id'], 'Which outside report is still missing?').json()
        history = [
            {"role": "user", "content": "Which outside report is still missing?"},
            {"role": "assistant", "content": first['answer']},
        ]
        second = ask(client, patient['id'], 'so nothing is missing?', history)
        assert second.status_code == 200
        data = second.json()
        assert data['answer_origin'] == 'provenance_fact_store'
        assert data['resolved_intent'] == 'pending_items'
        assert data['history_used'] is True
        assert data['answer'].startswith('No.')
        assert 'outside cardiology report' in data['answer'].casefold()
        assert model.calls == []
    finally:
        client.__exit__(None, None, None)


def test_condition_summary_answers_condition_not_first_source_sentence(tmp_path):
    model = NaturalConversationModel(); client, patient = open_workspace(tmp_path, model)
    try:
        data = ask(client, patient['id'], 'summarize the condition of the patient').json()
        assert data['resolved_intent'] == 'conditions'
        assert data['answer_origin'] == 'provenance_fact_store'
        answer = data['answer'].casefold()
        assert 'essential hypertension' in answer
        assert 'seasonal allergic rhinitis' in answer
        assert 'no interval changes since the prior records review' not in answer
        assert model.calls == []
    finally:
        client.__exit__(None, None, None)


def test_latest_report_summary_uses_one_latest_document(tmp_path):
    model = NaturalConversationModel(); client, patient = open_workspace(tmp_path, model)
    try:
        data = ask(client, patient['id'], 'summarize the latest report').json()
        assert data['resolved_intent'] == 'latest_report'
        assert data['answer_origin'] == 'structured_chart'
        assert len(data['sources']) == 1
        answer = data['answer'].casefold()
        assert 'october 1, 2026' in answer
        assert 'outside records review' in answer
        assert 'lisinopril' in answer
        assert 'cardiology report' in answer
        assert all(term not in answer for term in ['demo data', 'fictional', 'mrn', 'synthetic'])
    finally:
        client.__exit__(None, None, None)


def test_medication_wording_variants_return_only_current_medications(tmp_path):
    questions = [
        'what medications are prescribed to the patient?',
        'what medicines are the patient using?',
        "i'm asking for medication",
        'Which medications are documented in the chart?',
    ]
    for index, question in enumerate(questions):
        model = NaturalConversationModel(); client, patient = open_workspace(tmp_path / str(index), model)
        try:
            data = ask(client, patient['id'], question).json()
            assert data['resolved_intent'] == 'medications'
            assert data['answer_origin'] == 'provenance_fact_store'
            assert data['model_used'] is False and model.calls == []
            answer = data['answer'].casefold()
            assert 'lisinopril 10 mg' in answer and 'cetirizine 10 mg' in answer
            assert all(term not in answer for term in ['blood pressure', 'patient:', 'dob:', 'mrn:', 'demo data'])
            assert len(answer.split()) < 35
        finally:
            client.__exit__(None, None, None)


def test_retrieval_deduplicates_and_sanitizes_assistant_sources(tmp_path):
    model = NaturalConversationModel(); client, patient = open_workspace(tmp_path, model)
    try:
        data = ask(client, patient['id'], 'Which medications are documented?').json()
        assert 1 <= len(data['sources']) <= 2
        assert len({source['id'] for source in data['sources']}) == len(data['sources'])
        excerpts = '\n'.join(source['excerpt'] for source in data['sources']).casefold()
        assert all(term not in excerpts for term in ['demo data', 'fictional patient', 'patient:', 'dob:', 'mrn:'])
    finally:
        client.__exit__(None, None, None)


def test_pending_follow_up_uses_history_and_answers_missing_date_directly(tmp_path):
    model = NaturalConversationModel(); client, patient = open_workspace(tmp_path, model)
    try:
        first = ask(client, patient['id'], 'Which report is pending?')
        assert first.status_code == 200 and model.calls == []
        history = [
            {"role": "user", "content": "Which report is pending?"},
            {"role": "assistant", "content": first.json()['answer']},
        ]
        second = ask(client, patient['id'], 'When was it requested?', history)
        assert second.status_code == 200, second.text
        data = second.json()
        assert data['history_used'] is True
        assert data['resolved_intent'] == 'pending_items'
        assert data['answer_origin'] == 'structured_chart'
        assert 'cardiology report' in data['answer'].casefold()
        assert 'does not document when' in data['answer'].casefold()
        assert model.calls == []
    finally:
        client.__exit__(None, None, None)


def test_direct_treatment_question_is_answered_without_model(tmp_path):
    model = NaturalConversationModel(); client, patient = open_workspace(tmp_path, model)
    try:
        response = ask(client, patient['id'], 'What should I take for a fever?')
        assert response.status_code == 200
        data = response.json()
        assert data['answer_origin'] == 'safety_boundary'
        assert data['model_used'] is False
        assert data['sources'] == []
        assert 'can’t choose a medication' in data['answer']
        assert 'paracetamol' not in data['answer'].casefold()
        assert model.calls == []
    finally:
        client.__exit__(None, None, None)


def test_treatment_follow_up_uses_history_and_stays_inside_boundary(tmp_path):
    model = NaturalConversationModel(); client, patient = open_workspace(tmp_path, model)
    try:
        history = [
            {"role": "user", "content": "What should I take for a fever?"},
            {"role": "assistant", "content": "I can summarize the chart but cannot choose treatment."},
        ]
        response = ask(client, patient['id'], 'What other options are there?', history)
        assert response.status_code == 200
        data = response.json()
        assert data['answer_origin'] == 'safety_boundary'
        assert data['history_used'] is True
        assert model.calls == []
    finally:
        client.__exit__(None, None, None)


def test_unsafe_model_recommendation_for_open_question_is_replaced(tmp_path):
    model = UnsafeRecommendationModel(); client, patient = open_workspace(tmp_path, model)
    try:
        response = ask(client, patient['id'], 'Explain how the reviewed findings relate to one another.')
        assert response.status_code == 200
        data = response.json()
        assert data['answer_origin'] == 'record_fallback'
        assert model.calls
        assert 'paracetamol' not in data['answer'].casefold()
        assert 'should take' not in data['answer'].casefold()
        assert 'hypertension' in data['answer'].casefold()
    finally:
        client.__exit__(None, None, None)


def test_raw_record_dump_from_model_is_replaced_and_never_shown(tmp_path):
    model = RawDumpModel(); client, patient = open_workspace(tmp_path, model)
    try:
        data = ask(client, patient['id'], 'Explain the reviewed findings in plain language.').json()
        assert data['answer_origin'] == 'record_fallback'
        assert model.calls
        answer = data['answer'].casefold()
        assert 'hypertension' in answer
        assert all(term not in answer for term in ['demo data', 'fictional patient', 'patient:', 'dob:', 'mrn:'])
        assert len(data['answer']) < 1000
    finally:
        client.__exit__(None, None, None)


def test_requested_twenty_word_summary_is_bounded_and_clean(tmp_path):
    model = EchoOnlyModel(); client, patient = open_workspace(tmp_path, model)
    try:
        response = ask(client, patient['id'], 'Scan the available record and summarize it in 20 words.')
        assert response.status_code == 200
        data = response.json()
        words = data['answer'].split()
        assert 5 <= len(words) <= 20
        assert data['answer_origin'] == 'provenance_fact_store'
        assert model.calls == []
        assert 'synthetic' not in data['answer'].casefold()
        assert 'fictional' not in data['answer'].casefold()
        assert 'excerpt' not in data['answer'].casefold()
    finally:
        client.__exit__(None, None, None)


def test_conversation_history_is_bounded_by_schema(tmp_path):
    model = NaturalConversationModel(); client, patient = open_workspace(tmp_path, model)
    try:
        too_many = [{"role": "user", "content": f"turn {index}"} for index in range(9)]
        response = ask(client, patient['id'], 'What is documented?', too_many)
        assert response.status_code == 422
        assert model.calls == []
    finally:
        client.__exit__(None, None, None)


def test_demo_seed_records_are_detailed_for_all_patients(tmp_path):
    model = NaturalConversationModel(); client, patient = open_workspace(tmp_path, model)
    try:
        patients = client.get('/api/patients').json()
        assert len(patients) == 3
        for row in patients:
            detail = client.get(f"/api/patients/{row['id']}").json()
            assert len(detail['documents']) >= 1
            assert len(detail['notes']) >= 1
            combined = ' '.join(note['body'] for note in detail['notes']).casefold()
            assert 'active problems:' in combined
            assert 'current medications:' in combined
            assert 'uploaded history is incomplete' not in combined
    finally:
        client.__exit__(None, None, None)


def test_reported_multi_source_demo_flow_remains_relevant_and_concise(tmp_path):
    """Reproduce the user's chart with legacy, update, note, and detailed sources."""
    model = NaturalConversationModel(); client, patient = open_workspace(tmp_path, model)
    try:
        # Add the old sparse note that existed in earlier demo workspaces.
        assert client.post(f"/api/patients/{patient['id']}/notes", json={
            "title": "Records review",
            "body": "Synthetic training record. Uploaded history is incomplete. Confirm original source documentation at the next visit.",
        }).status_code == 201

        update_text = (
            "SYNTHETIC TRAINING RECORD — NOT A REAL PATIENT\n"
            "Patient: Morgan Avery\nDOB: 1988-04-12\nMRN: SYN-1001\n"
            "Report date: 2026-10-04\n\nRecord update:\n"
            "The patient reports no interval changes since the prior records review.\n"
            "An outside report has been requested but has not yet been received.\n"
            "No new laboratory values, medication changes, or treatment decisions are documented."
        )
        upload = client.post('/api/intake', content=update_text.encode(), headers={
            'X-Filename': 'morgan-update.txt', 'X-Use-Model': 'false',
        })
        assert upload.status_code == 201
        document = upload.json()['document']
        review = client.post(f"/api/documents/{document['id']}/review", json={
            'patient_id': patient['id'], 'text': document['text'],
            'summary': 'No interval changes; an outside report remains outstanding.',
            'expected_version': document['version'], 'reason': 'Reviewed synthetic update',
            'identity_confirmed': True, 'content_confirmed': True,
        })
        assert review.status_code == 200

        pending = ask(client, patient['id'], 'Which outside report is still missing?').json()
        assert 'outside cardiology report' in pending['answer'].casefold()
        assert 'no interval changes' not in pending['answer'].casefold()
        assert len(pending['sources']) <= 2

        condition = ask(client, patient['id'], 'summarize the condition of the patient').json()
        assert 'essential hypertension' in condition['answer'].casefold()
        assert 'seasonal allergic rhinitis' in condition['answer'].casefold()

        medication = ask(client, patient['id'], 'what medicines are the patient using?').json()
        assert 'lisinopril 10 mg' in medication['answer'].casefold()
        assert 'cetirizine 10 mg' in medication['answer'].casefold()
        assert len(medication['answer'].split()) < 35

        latest = ask(client, patient['id'], 'summarize the latest report').json()
        assert 'october 4, 2026' in latest['answer'].casefold()
        assert 'no interval changes' in latest['answer'].casefold()
        assert 'outside report' in latest['answer'].casefold()
        assert len(latest['sources']) == 1

        combined = json.dumps([pending, condition, medication, latest]).casefold()
        assert all(term not in combined for term in ['demo data', 'fictional patient', 'mrn: syn-1001'])
        assert model.calls == []
    finally:
        client.__exit__(None, None, None)


class RuntimeMetricsModel(NaturalConversationModel):
    def last_request_metrics(self):
        return {
            "context_tokens": 4096,
            "prompt_tokens": 700,
            "prompt_limit_tokens": 3276,
            "prompt_fraction": 0.1709,
            "output_reserve_tokens": 650,
            "safety_margin_tokens": 64,
            "headroom_tokens": 2746,
            "count_method": "llama_cpp_input_tokens",
            "within_budget": True,
            "history_turns_available": 0,
            "history_turns_supplied": 0,
            "source_blocks_available": 1,
            "source_blocks_supplied": 1,
        }


def test_open_model_answer_returns_non_phi_runtime_metrics(tmp_path):
    model = RuntimeMetricsModel(); client, patient = open_workspace(tmp_path, model)
    try:
        response = ask(client, patient['id'], 'How do the reviewed findings relate to each other?')
        assert response.status_code == 200, response.text
        data = response.json()
        assert data['answer_origin'] == 'model'
        assert data['model_runtime']['count_method'] == 'llama_cpp_input_tokens'
        assert data['model_runtime']['prompt_fraction'] <= 0.80
        encoded = json.dumps(data['model_runtime']).casefold()
        assert patient['name'].casefold() not in encoded
        assert patient['mrn'].casefold() not in encoded
        audit = client.get(f"/api/patients/{patient['id']}/audit").json()
        completed = next(event for event in audit if event['action'] == 'assistant.completed')
        assert completed['details']['model_runtime']['within_budget'] is True
    finally:
        client.__exit__(None, None, None)
