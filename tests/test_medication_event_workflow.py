"""Medication-event retrieval integrated into the encrypted chart assistant."""
from __future__ import annotations

import sqlite3

import pytest

import app.main as main_module
from app.medication_event_workflow import (
    EligibleSource,
    _rank_sources_fts,
    load_eligible_sources,
    search_medication_event,
    validate_source_snapshot,
)
from tests.test_assistant_threads import (
    USER_PASS,
    ask,
    create_user,
    grant,
    open_thread,
    workspace,  # noqa: F401
)


def add_note(client, patient_id: str, title: str, body: str):
    response = client.post(f"/api/patients/{patient_id}/notes", json={"title": title, "body": body})
    assert response.status_code == 201, response.text
    return response.json()


def reopen_thread(client, patient_id: str):
    response = open_thread(client, patient_id)
    assert response.status_code == 200, response.text
    return response.json()


def test_default_morgan_prescription_date_reports_absence_not_report_date(workspace):
    client, app, model, morgan, *_ = workspace
    thread = reopen_thread(client, morgan['id'])
    response = ask(client, morgan['id'], thread['id'], 'What date was Morgan prescribed lisinopril?')
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['answer_origin'] == 'workflow_search'
    assert data['model_used'] is False
    assert data['workflow_search']['status'] == 'no_explicit_date_found'
    assert data['workflow_search']['event_type'] == 'prescribed'
    assert data['workflow_search']['medications'] == ['lisinopril']
    assert data['workflow_search']['searched_source_count'] == data['workflow_search']['eligible_source_count'] == 2
    assert 'report dates were not substituted' in data['answer'].casefold()
    assert '2026-10-01' not in data['answer']
    assert '2026-09-28' not in data['answer']
    assert model.calls == []


def test_explicit_prescribed_and_dispensed_dates_remain_distinct(workspace):
    client, app, model, morgan, *_ = workspace
    add_note(
        client,
        morgan['id'],
        'Medication event timeline',
        'Lisinopril was prescribed on 2026-09-04.\nLisinopril was dispensed on 2026-09-06.',
    )
    thread = reopen_thread(client, morgan['id'])

    prescribed = ask(client, morgan['id'], thread['id'], 'When was lisinopril prescribed?').json()
    assert prescribed['answer_origin'] == 'workflow_search'
    assert prescribed['workflow_search']['status'] == 'documented_in_source'
    assert '2026-09-04' in prescribed['answer']
    assert '2026-09-06' not in prescribed['answer']
    assert prescribed['workflow_search']['citations'][0]['quote'] == 'Lisinopril was prescribed on 2026-09-04.'

    dispensed = ask(client, morgan['id'], thread['id'], 'When was lisinopril dispensed?').json()
    assert dispensed['answer_origin'] == 'workflow_search'
    assert '2026-09-06' in dispensed['answer']
    assert '2026-09-04' not in dispensed['answer']
    assert dispensed['workflow_search']['event_type'] == 'dispensed'
    assert model.calls == []


def test_unrelated_report_or_order_date_is_not_misattributed_to_lisinopril(workspace):
    client, app, model, morgan, *_ = workspace
    add_note(
        client,
        morgan['id'],
        'Ambiguous date fixture',
        'Lisinopril was listed in the medication record. The cardiology report was ordered on 2026-09-04.',
    )
    thread = reopen_thread(client, morgan['id'])
    data = ask(client, morgan['id'], thread['id'], 'When was lisinopril prescribed?').json()
    assert data['answer_origin'] == 'workflow_search'
    assert data['workflow_search']['status'] in {'no_explicit_date_found', 'needs_review'}
    assert '2026-09-04' not in data['answer']
    assert model.calls == []


def test_received_medication_requires_event_clarification_and_selection(workspace):
    client, app, model, morgan, *_ = workspace
    add_note(
        client,
        morgan['id'],
        'Dispensing fixture',
        'Lisinopril was dispensed on 2026-09-06.',
    )
    thread = reopen_thread(client, morgan['id'])

    clarification = ask(client, morgan['id'], thread['id'], 'When was lisinopril received?').json()
    assert clarification['answer_origin'] == 'clarification'
    assert clarification['workflow_search']['status'] == 'needs_event'
    assert 'dispensed' in clarification['answer'].casefold()
    assert 'picked up' in clarification['answer'].casefold()
    state = clarification['dialogue_state']
    assert state['relation'] == 'medication_event_date'
    assert state['clarification']['required'] is True

    resolved = ask(client, morgan['id'], thread['id'], 'dispensed').json()
    assert resolved['answer_origin'] == 'workflow_search'
    assert resolved['workflow_search']['event_type'] == 'dispensed'
    assert resolved['workflow_search']['medications'] == ['lisinopril']
    assert '2026-09-06' in resolved['answer']
    assert model.calls == []


def test_exact_failed_conversation_preserves_date_request_through_both(workspace):
    client, app, model, morgan, *_ = workspace
    thread = reopen_thread(client, morgan['id'])
    meds = ask(client, morgan['id'], thread['id'], 'what meds is morgan taking?').json()
    assert meds['answer_origin'] == 'provenance_fact_store'

    clarification = ask(client, morgan['id'], thread['id'], 'when was it requested?').json()
    assert clarification['answer_origin'] == 'clarification'
    assert clarification['workflow_search']['status'] == 'needs_medication'

    both = ask(client, morgan['id'], thread['id'], 'both').json()
    assert both['answer_origin'] == 'workflow_search'
    assert both['workflow_search']['event_type'] == 'prescribed'
    assert both['workflow_search']['medications'] == ['lisinopril', 'cetirizine']
    assert 'lisinopril' in both['answer'].casefold()
    assert 'cetirizine' in both['answer'].casefold()
    assert 'no explicit prescribed date' in both['answer'].casefold()
    assert both['dialogue_state']['relation'] == 'prescribed_date'
    assert model.calls == []


def test_named_medication_after_clarification_preserves_prescription_date_task(workspace):
    client, app, model, morgan, *_ = workspace
    thread = reopen_thread(client, morgan['id'])
    ask(client, morgan['id'], thread['id'], 'What medications are documented?')
    ask(client, morgan['id'], thread['id'], 'When was it requested?')
    result = ask(client, morgan['id'], thread['id'], 'lisinopril').json()
    assert result['answer_origin'] == 'workflow_search'
    assert result['workflow_search']['event_type'] == 'prescribed'
    assert result['workflow_search']['medications'] == ['lisinopril']
    assert 'no explicit prescribed date' in result['answer'].casefold()


def test_other_patient_event_never_leaks_into_morgan(workspace):
    client, app, model, morgan, jordan, *_ = workspace
    add_note(
        client,
        jordan['id'],
        'Other patient medication event',
        'Lisinopril was prescribed on 2026-05-05.',
    )
    thread = reopen_thread(client, morgan['id'])
    data = ask(client, morgan['id'], thread['id'], 'When was lisinopril prescribed?').json()
    assert '2026-05-05' not in data['answer']
    assert all('2026-05-05' not in citation['quote'] for citation in data['workflow_search']['citations'])
    assert data['workflow_search']['eligible_source_count'] == 2


def test_unassigned_user_cannot_use_event_search(workspace):
    client, app, model, morgan, *_ = workspace
    user, headers = create_user(client, 'doctor', 'event-search-doctor')
    # No grant yet: patient and thread are unavailable.
    detail = client.get(f"/api/patients/{morgan['id']}", headers=headers)
    assert detail.status_code == 404
    grant(client, morgan['id'], user['id'])
    thread = open_thread(client, morgan['id'], headers=headers).json()
    detail = client.get(f"/api/patients/{morgan['id']}", headers=headers).json()
    allowed = ask(client, morgan['id'], thread['id'], 'When was lisinopril prescribed?', headers=headers)
    assert allowed.status_code == 200
    revoke = client.post(f"/api/patients/{morgan['id']}/access", json={
        'user_id': user['id'], 'enabled': False, 'reason': 'End synthetic assignment',
    })
    assert revoke.status_code == 200
    blocked = client.post(
        f"/api/patients/{morgan['id']}/ask",
        json={
            'question': 'When was lisinopril prescribed?',
            'expected_revision': detail['patient']['revision'],
            'thread_id': thread['id'],
            'history': [],
        },
        headers=headers,
    )
    assert blocked.status_code == 404


def test_source_snapshot_recheck_rejects_non_revision_source_mutation(workspace, monkeypatch):
    client, app, model, morgan, *_ = workspace
    thread = reopen_thread(client, morgan['id'])
    original = main_module.search_medication_event

    def mutate_after_search(db, **kwargs):
        result = original(db, **kwargs)
        if result is not None:
            note = db.execute(
                'SELECT id FROM notes WHERE patient_id=? AND context_enabled=1 LIMIT 1',
                (morgan['id'],),
            ).fetchone()
            if note:
                db.execute('UPDATE notes SET context_enabled=0 WHERE id=?', (note['id'],))
        return result

    monkeypatch.setattr(main_module, 'search_medication_event', mutate_after_search)
    response = ask(client, morgan['id'], thread['id'], 'When was lisinopril prescribed?')
    assert response.status_code == 409
    assert 'sources changed while searching' in response.text.casefold()


def test_scoped_fts_handles_query_metacharacters_without_sql_or_fts_injection():
    sources = [
        EligibleSource('1', 'N', 'Medication note', 'Lisinopril was prescribed on 2026-09-04.', 1, None, None),
        EligibleSource('2', 'N', 'Other note', 'Cetirizine 10 mg is current.', 1, None, None),
    ]
    ids = _rank_sources_fts(sources, 'lisinopril OR 1=1; DROP TABLE scoped_fts --')
    assert set(ids) <= {'1', '2'}


def test_snapshot_detects_version_or_content_change():
    before = [EligibleSource('1', 'N', 'Note', 'Lisinopril was prescribed on 2026-09-04.', 1, None, None)]
    expected = {'1': before[0].snapshot()}
    assert validate_source_snapshot(expected, before)
    version_changed = [EligibleSource('1', 'N', 'Note', before[0].body, 2, None, None)]
    content_changed = [EligibleSource('1', 'N', 'Note', 'Lisinopril was prescribed on 2026-09-05.', 1, None, None)]
    assert not validate_source_snapshot(expected, version_changed)
    assert not validate_source_snapshot(expected, content_changed)


def test_search_budget_fails_incomplete_not_absent():
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.executescript('''
        CREATE TABLE documents(id TEXT,patient_id TEXT,filename TEXT,text TEXT,version INTEGER,updated_at TEXT,state TEXT,context_enabled INTEGER);
        CREATE TABLE notes(id TEXT,patient_id TEXT,title TEXT,body TEXT,version INTEGER,updated_at TEXT,context_enabled INTEGER);
    ''')
    for index in range(257):
        conn.execute(
            'INSERT INTO notes VALUES(?,?,?,?,?,?,1)',
            (str(index), 'p', f'n{index}', 'Current medications: lisinopril 10 mg daily.', 1, '2026-01-01'),
        )
    class R:
        intent = 'medications'; relation = 'new_question'; referent = {}
    result = search_medication_event(
        conn, patient_id='p', question='When was lisinopril prescribed?', resolution=R(), history=[]
    )
    assert result is not None
    assert result.status == 'source_budget_exceeded'
    assert 'can’t claim that the date is absent' in result.answer
