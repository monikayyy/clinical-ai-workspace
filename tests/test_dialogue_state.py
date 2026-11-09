"""Increment 3: deterministic structured dialogue-state resolution."""
from __future__ import annotations

import pytest

from tests.test_assistant_threads import ask, open_thread, workspace  # noqa: F401


def _thread_state(client, thread_id):
    return client.get(f"/api/assistant/threads/{thread_id}").json()["dialogue_state"]


def test_status_reports_dialogue_state_revision(workspace):
    client, *_ = workspace
    status = client.get('/api/status').json()
    assert status['assistant_revision'] == 'dialogue-state-v1'
    assert status['stage2_increment'] == 'provenance-conflict-semantics'
    assert status['schema_version'] == 5  # Provenance conflict/lifecycle schema.


def test_pending_referent_is_persisted_and_resolves_request_date(workspace):
    client, app, model, morgan, *_ = workspace
    thread = open_thread(client, morgan['id']).json()
    first = ask(client, morgan['id'], thread['id'], 'Which outside report is still missing?')
    assert first.status_code == 200
    data = first.json()
    assert data['answer_origin'] == 'provenance_fact_store'
    state = data['dialogue_state']
    assert state['active_intent'] == 'pending_items'
    assert state['referent']['type'] == 'pending_item'
    assert 'cardiology report' in state['referent']['selected']['label'].casefold()
    assert state['resolver_confidence'] >= 0.95

    follow = ask(client, morgan['id'], thread['id'], 'When was it requested?')
    assert follow.status_code == 200
    result = follow.json()
    assert result['answer_origin'] == 'structured_chart'
    assert 'does not document when it was requested' in result['answer'].casefold()
    assert result['dialogue_state']['relation'] == 'request_date'
    assert result['dialogue_state']['referent']['selected']['label'] == state['referent']['selected']['label']
    assert model.calls == []


def test_received_status_follow_up_uses_saved_referent(workspace):
    client, app, model, morgan, *_ = workspace
    thread = open_thread(client, morgan['id']).json()
    ask(client, morgan['id'], thread['id'], 'Which report is pending?')
    response = ask(client, morgan['id'], thread['id'], 'Was it ever received?')
    assert response.status_code == 200
    data = response.json()
    assert data['answer_origin'] == 'structured_chart'
    assert 'still lists the outside cardiology report as pending' in data['answer'].casefold()
    assert 'later reviewed entry' in data['answer'].casefold()
    assert data['dialogue_state']['relation'] == 'received_status'
    assert model.calls == []


def test_multiple_medications_trigger_clarification_then_specific_dose(workspace):
    client, app, model, morgan, *_ = workspace
    thread = open_thread(client, morgan['id']).json()
    medications = ask(client, morgan['id'], thread['id'], 'What medications are documented?')
    assert medications.status_code == 200
    med_state = medications.json()['dialogue_state']
    labels = [item['label'].casefold() for item in med_state['referent']['items']]
    assert any('lisinopril' in label for label in labels)
    assert any('cetirizine' in label for label in labels)
    assert 'selected' not in med_state['referent']

    clarification = ask(client, morgan['id'], thread['id'], 'What dose?')
    assert clarification.status_code == 200
    clarify_data = clarification.json()
    assert clarify_data['answer_origin'] == 'clarification'
    assert clarify_data['clarification_required'] is True
    assert 'lisinopril' in clarify_data['answer'].casefold()
    assert 'cetirizine' in clarify_data['answer'].casefold()
    assert clarify_data['sources'] == []
    assert model.calls == []

    selected = ask(client, morgan['id'], thread['id'], 'Lisinopril')
    assert selected.status_code == 200
    selected_data = selected.json()
    assert selected_data['answer_origin'] == 'structured_chart'
    assert 'lisinopril' in selected_data['answer'].casefold()
    assert '10 mg' in selected_data['answer'].casefold()
    assert selected_data['clarification_required'] is False
    assert selected_data['dialogue_state']['relation'] == 'dose'
    assert selected_data['dialogue_state']['referent']['selected']['label'].casefold() == 'lisinopril'
    assert model.calls == []


def test_ordinal_clarification_selection_and_frequency(workspace):
    client, app, model, morgan, *_ = workspace
    thread = open_thread(client, morgan['id']).json()
    ask(client, morgan['id'], thread['id'], 'List the medications in the chart.')
    clarify = ask(client, morgan['id'], thread['id'], 'How often?')
    assert clarify.json()['answer_origin'] == 'clarification'
    selected = ask(client, morgan['id'], thread['id'], 'The second one')
    data = selected.json()
    assert data['answer_origin'] == 'structured_chart'
    assert 'cetirizine' in data['answer'].casefold()
    assert 'as needed' in data['answer'].casefold()
    assert data['dialogue_state']['relation'] == 'frequency'
    assert model.calls == []


def test_unresolved_clarification_repeats_without_model_call(workspace):
    client, app, model, morgan, *_ = workspace
    thread = open_thread(client, morgan['id']).json()
    ask(client, morgan['id'], thread['id'], 'What medications are documented?')
    first = ask(client, morgan['id'], thread['id'], 'What dose?').json()
    second = ask(client, morgan['id'], thread['id'], 'I am not sure').json()
    assert second['answer_origin'] == 'clarification'
    assert second['answer'] == first['answer']
    assert second['dialogue_state']['clarification']['required'] is True
    assert model.calls == []


def test_vague_question_without_state_requests_topic_clarification(workspace):
    client, app, model, morgan, *_ = workspace
    thread = open_thread(client, morgan['id']).json()
    response = ask(client, morgan['id'], thread['id'], 'What dose?')
    assert response.status_code == 200
    data = response.json()
    assert data['answer_origin'] == 'clarification'
    assert 'what would you like me to check' in data['answer'].casefold()
    assert data['resolver_confidence'] <= 0.25
    assert data['dialogue_state']['active_intent'] == 'open_question'
    assert model.calls == []


def test_treatment_safety_state_persists_then_explicit_chart_query_resets(workspace):
    client, app, model, morgan, *_ = workspace
    thread = open_thread(client, morgan['id']).json()
    first = ask(client, morgan['id'], thread['id'], 'What should I take for a fever?').json()
    assert first['answer_origin'] == 'safety_boundary'
    assert first['dialogue_state']['safety_mode'] == 'treatment_boundary'

    follow = ask(client, morgan['id'], thread['id'], 'What other options are there?').json()
    assert follow['answer_origin'] == 'safety_boundary'
    assert follow['dialogue_state']['safety_mode'] == 'treatment_boundary'

    chart = ask(client, morgan['id'], thread['id'], 'What medications are documented in the chart?').json()
    assert chart['answer_origin'] == 'provenance_fact_store'
    assert chart['dialogue_state']['active_intent'] == 'medications'
    assert chart['dialogue_state']['safety_mode'] == 'chart_only'
    assert 'lisinopril' in chart['answer'].casefold()
    assert model.calls == []


def test_requested_format_is_persisted_and_follow_up_can_shorten(workspace):
    client, app, model, morgan, *_ = workspace
    thread = open_thread(client, morgan['id']).json()
    listed = ask(client, morgan['id'], thread['id'], 'List the medications in bullet points.').json()
    assert listed['answer_origin'] == 'provenance_fact_store'
    assert '- ' in listed['answer']
    assert listed['dialogue_state']['requested_format']['style'] == 'bullets'

    shorter = ask(client, morgan['id'], thread['id'], 'Make it shorter.').json()
    assert shorter['answer_origin'] == 'provenance_fact_store'
    assert shorter['dialogue_state']['active_intent'] == 'medications'
    assert shorter['dialogue_state']['requested_format']['style'] == 'brief'
    assert len(shorter['answer'].split()) <= 35
    assert model.calls == []


def test_latest_report_follow_up_checks_for_later_report(workspace):
    client, app, model, morgan, *_ = workspace
    thread = open_thread(client, morgan['id']).json()
    latest = ask(client, morgan['id'], thread['id'], 'Summarize the latest report.').json()
    assert latest['answer_origin'] == 'structured_chart'
    assert latest['dialogue_state']['referent']['type'] == 'report'
    follow = ask(client, morgan['id'], thread['id'], 'What changed after that?').json()
    assert follow['answer_origin'] == 'structured_chart'
    assert 'don’t see a later reviewed report' in follow['answer'].casefold()
    assert follow['dialogue_state']['relation'] == 'changed_after'
    assert model.calls == []


def test_chart_revision_reset_clears_referent_and_clarification(workspace):
    client, app, model, morgan, *_ = workspace
    thread = open_thread(client, morgan['id']).json()
    ask(client, morgan['id'], thread['id'], 'What medications are documented?')
    ask(client, morgan['id'], thread['id'], 'What dose?')
    before = _thread_state(client, thread['id'])
    assert before['clarification']['required'] is True

    created = client.post(f"/api/patients/{morgan['id']}/notes", json={
        "title": "Synthetic revision note",
        "body": "A reviewed synthetic note that changes the patient chart revision.",
    })
    assert created.status_code == 201
    reopened = open_thread(client, morgan['id']).json()
    assert reopened['context_reset'] is True
    after = reopened['dialogue_state']
    assert after['active_intent'] is None
    assert after['referent'] == {}
    assert after['clarification']['required'] is False
    assert after['resolver_confidence'] == 0.0


def test_dialogue_state_survives_logout_and_login(workspace):
    client, app, model, morgan, *_ = workspace
    thread = open_thread(client, morgan['id']).json()
    ask(client, morgan['id'], thread['id'], 'Which report is pending?')
    before = _thread_state(client, thread['id'])
    assert before['referent']['selected']['label']
    assert client.post('/api/logout').status_code == 200
    login = client.post('/api/login', json={
        "username": "admin",
        "password": "assistant-thread-admin-password",
    }).json()
    client.headers['Authorization'] = 'Bearer ' + login['token']
    resumed = open_thread(client, morgan['id']).json()
    after = resumed['dialogue_state']
    assert after['active_intent'] == before['active_intent']
    assert after['referent']['selected']['label'] == before['referent']['selected']['label']
    assert after['resolver_confidence'] == before['resolver_confidence']
