"""Conflict, correction, renewal, lifecycle, and effective-time regressions."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.fact_store import (
    EXTRACTION_VERSION,
    SEMANTIC_VERSION,
    fact_resolutions_for_patient,
    facts_for_patient,
    medication_event_resolution,
    medication_state_resolution,
)
from app.vault import (
    FACT_SEMANTICS_DESCRIPTION,
    PROVENANCE_FACT_DESCRIPTION,
    PROVENANCE_FACT_VERSION,
    SCHEMA_VERSION,
    Vault,
    now,
)
from tests.test_assistant_threads import ADMIN_PASS, VAULT_PASS, create_user, grant, workspace  # noqa: F401
from tests.test_medication_event_workflow import add_note, ask, reopen_thread


def resolutions(client, patient_id, **params):
    response = client.get(f"/api/patients/{patient_id}/facts/resolutions", params=params)
    assert response.status_code == 200, response.text
    return response.json()["resolutions"]


def test_schema5_has_semantic_columns_and_resolution_table(workspace):
    client, app, model, morgan, *_ = workspace
    status = client.get('/api/status').json()
    assert status['schema_version'] == SCHEMA_VERSION == 5
    assert status['stage2_increment'] == 'provenance-conflict-semantics'
    assert status['workflow_revision'] == EXTRACTION_VERSION
    with app.state.vault.mutex:
        db = app.state.vault.db
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert 'clinical_fact_resolutions' in tables
        columns = {row[1] for row in db.execute('PRAGMA table_info(clinical_facts)')}
        assert {
            'event_role','asserted_at','effective_start','effective_end',
            'assertion_key','supersedes_json','semantic_version'
        } <= columns
        migration = db.execute('SELECT description FROM schema_migrations WHERE version=5').fetchone()[0]
        assert migration == FACT_SEMANTICS_DESCRIPTION
    inventory = client.get(f"/api/patients/{morgan['id']}/facts").json()['inventory']
    assert inventory['extraction_version'] == EXTRACTION_VERSION
    assert inventory['semantic_version'] == SEMANTIC_VERSION
    assert inventory['resolution_count'] > 0


def _downgrade_to_real_v4(vault: Vault) -> None:
    """Turn a fresh v5 fixture into the exact v4 derived-fact shape."""
    with vault.transaction() as db:
        # Preserve a synthetic derived row to prove the v5 migration discards
        # only rebuildable cache rows, not chart/thread data.
        db.execute('DROP TABLE clinical_fact_resolutions')
        for index_name in [
            'clinical_facts_patient_type_idx','clinical_facts_patient_entity_idx',
            'clinical_facts_patient_event_idx','clinical_facts_source_idx',
            'clinical_facts_assertion_idx','clinical_facts_patient_event_role_idx',
        ]:
            db.execute(f'DROP INDEX IF EXISTS {index_name}')
        db.execute('ALTER TABLE clinical_facts RENAME TO clinical_facts_v5')
        db.execute('''CREATE TABLE clinical_facts (
             id TEXT PRIMARY KEY, patient_id TEXT NOT NULL REFERENCES patients(id), session_id TEXT REFERENCES sessions(id),
             fact_type TEXT NOT NULL, entity_key TEXT, entity_label TEXT, event_type TEXT,
             value_text TEXT NOT NULL, normalized_json TEXT NOT NULL DEFAULT '{}', event_date TEXT,
             status TEXT NOT NULL DEFAULT 'current', polarity TEXT NOT NULL DEFAULT 'affirmed',
             certainty TEXT NOT NULL DEFAULT 'explicit', source_kind TEXT NOT NULL CHECK(source_kind IN ('document','note')),
             source_id TEXT NOT NULL, source_version INTEGER NOT NULL, source_sha256 TEXT NOT NULL,
             source_title TEXT NOT NULL, quote TEXT NOT NULL, start_offset INTEGER NOT NULL, end_offset INTEGER NOT NULL,
             extraction_method TEXT NOT NULL, extraction_version TEXT NOT NULL, created_at TEXT NOT NULL,
             UNIQUE(source_kind,source_id,source_version,fact_type,entity_key,event_type,start_offset,end_offset,value_text)
        )''')
        db.execute('CREATE INDEX clinical_facts_patient_type_idx ON clinical_facts(patient_id,fact_type)')
        db.execute('CREATE INDEX clinical_facts_patient_entity_idx ON clinical_facts(patient_id,entity_key)')
        db.execute('CREATE INDEX clinical_facts_patient_event_idx ON clinical_facts(patient_id,event_type,event_date)')
        db.execute('CREATE INDEX clinical_facts_source_idx ON clinical_facts(source_kind,source_id,source_version)')
        db.execute('''INSERT INTO clinical_facts(
            id,patient_id,session_id,fact_type,entity_key,entity_label,event_type,value_text,normalized_json,event_date,
            status,polarity,certainty,source_kind,source_id,source_version,source_sha256,source_title,quote,start_offset,end_offset,
            extraction_method,extraction_version,created_at
        ) SELECT id,patient_id,session_id,fact_type,entity_key,entity_label,event_type,value_text,normalized_json,event_date,
            status,polarity,certainty,source_kind,source_id,source_version,source_sha256,source_title,quote,start_offset,end_offset,
            extraction_method,'provenance-facts-v1',created_at FROM clinical_facts_v5 LIMIT 1''')
        db.execute('DROP TABLE clinical_facts_v5')
        db.execute('DELETE FROM schema_migrations WHERE version=5')
        db.execute('PRAGMA user_version=4')


def test_version4_migration_preserves_chart_and_threads_but_rebuilds_derived_cache(tmp_path):
    from app.main import create_app
    from tests.test_assistant_threads import ThreadModel
    from fastapi.testclient import TestClient

    model = ThreadModel(); app = create_app(tmp_path, model)
    with TestClient(app) as client:
        assert client.post('/api/setup', json={
            'username':'admin','password':ADMIN_PASS,'vault_password':VAULT_PASS,
        }).status_code == 200
        token = client.post('/api/login', json={'username':'admin','password':ADMIN_PASS}).json()['token']
        client.headers['Authorization'] = 'Bearer ' + token
        assert client.post('/api/seed').status_code == 201
        patient = next(row for row in client.get('/api/patients').json() if row['mrn']=='SYN-1001')
        thread = reopen_thread(client, patient['id'])
        assert ask(client, patient['id'], thread['id'], 'What medications are documented?').status_code == 200
        # Materialize v5 facts before simulating a real v4 source vault.
        assert client.get(f"/api/patients/{patient['id']}/facts").status_code == 200
        thread_id = thread['id']
        before_messages = app.state.vault.db.execute('SELECT COUNT(*) FROM assistant_messages WHERE thread_id=?',(thread_id,)).fetchone()[0]
        _downgrade_to_real_v4(app.state.vault)
        app.state.vault.lock()

    migrated = Vault(tmp_path)
    migrated.unlock(VAULT_PASS)
    assert migrated.schema_version() == 5
    assert migrated.last_migrations == [
        f"Migrated vault to schema version 5: {FACT_SEMANTICS_DESCRIPTION}"
    ]
    assert migrated.db.execute("SELECT COUNT(*) FROM patients WHERE mrn='SYN-1001'").fetchone()[0] == 1
    assert migrated.db.execute('SELECT COUNT(*) FROM assistant_messages WHERE thread_id=?',(thread_id,)).fetchone()[0] == before_messages
    assert migrated.db.execute('SELECT COUNT(*) FROM clinical_facts').fetchone()[0] == 0
    assert migrated.db.execute('SELECT COUNT(*) FROM clinical_fact_sources').fetchone()[0] == 0
    assert migrated.db.execute('SELECT COUNT(*) FROM clinical_fact_resolutions').fetchone()[0] == 0
    assert migrated.verify_audit()['valid']
    migrated.close()


def test_original_prescription_and_renewal_are_separate_semantic_events(workspace):
    client, app, model, morgan, *_ = workspace
    add_note(client, morgan['id'], 'Medication orders',
             'Lisinopril was first prescribed on 2026-09-01.\nLisinopril was renewed on 2026-10-01.')
    original = resolutions(client, morgan['id'], fact_type='medication_event', entity='lisinopril', event_type='prescribed')
    renewal = resolutions(client, morgan['id'], fact_type='medication_event', entity='lisinopril', event_type='renewed')
    assert len(original) == 1 and original[0]['event_role'] == 'original'
    assert original[0]['values']['event_dates'] == ['2026-09-01']
    assert len(renewal) == 1 and renewal[0]['event_role'] == 'renewal'
    assert renewal[0]['values']['event_dates'] == ['2026-10-01']
    thread = reopen_thread(client, morgan['id'])
    first = ask(client, morgan['id'], thread['id'], 'When was lisinopril first prescribed?').json()
    assert '2026-09-01' in first['answer'] and '2026-10-01' not in first['answer']
    renewed = ask(client, morgan['id'], thread['id'], 'When was lisinopril renewed?').json()
    assert '2026-10-01' in renewed['answer'] and '2026-09-01' not in renewed['answer']


def test_conflicting_prescription_dates_are_not_silently_latest_wins(workspace):
    client, app, model, morgan, *_ = workspace
    add_note(client, morgan['id'], 'Prescription source A', 'Lisinopril was prescribed on 2026-09-01.')
    add_note(client, morgan['id'], 'Prescription source B', 'Lisinopril was prescribed on 2026-09-03.')
    rows = resolutions(client, morgan['id'], fact_type='medication_event', entity='lisinopril', event_type='prescribed')
    generic = next(row for row in rows if row['event_role'] == 'unspecified')
    assert generic['status'] == 'conflict'
    assert generic['preferred_fact_id'] is None
    assert generic['values']['event_dates'] == ['2026-09-01','2026-09-03']
    thread = reopen_thread(client, morgan['id'])
    result = ask(client, morgan['id'], thread['id'], 'When was lisinopril prescribed?').json()
    assert result['workflow_search']['status'] == 'conflicting_documentation'
    assert '2026-09-01' in result['answer'] and '2026-09-03' in result['answer']
    assert 'no date was selected as authoritative' in result['answer'].casefold()
    assert model.calls == []


def test_explicit_correction_supersedes_old_prescription_date(workspace):
    client, app, model, morgan, *_ = workspace
    old = add_note(client, morgan['id'], 'Original prescription entry', 'Lisinopril was prescribed on 2026-09-01.')
    correction = add_note(client, morgan['id'], 'Correction entry',
                          'Correction: Lisinopril was prescribed on 2026-09-03; prior date 2026-09-01 was incorrect.')
    rows = resolutions(client, morgan['id'], fact_type='medication_event', entity='lisinopril', event_type='prescribed')
    row = next(item for item in rows if item['event_role'] == 'unspecified')
    assert row['status'] in {'single','consistent'}
    assert row['values']['event_dates'] == ['2026-09-03']
    assert row['superseded_fact_ids']
    facts = client.get(f"/api/patients/{morgan['id']}/facts", params={'fact_type':'medication_event','entity':'lisinopril','event_type':'prescribed'}).json()['facts']
    by_id = {fact['id']: fact for fact in facts}
    assert any(by_id[fid]['source']['id'] == old['id'] for fid in row['superseded_fact_ids'])
    assert by_id[row['preferred_fact_id']]['source']['id'] == correction['id']
    thread = reopen_thread(client, morgan['id'])
    result = ask(client, morgan['id'], thread['id'], 'When was lisinopril prescribed?').json()
    assert '2026-09-03' in result['answer'] and '2026-09-01' not in result['answer']


def test_dated_discontinuation_resolves_medication_as_discontinued(workspace):
    client, app, model, morgan, *_ = workspace
    # Source record date makes the baseline current-med assertion older than the
    # explicit discontinuation event.
    add_note(client, morgan['id'], 'Medication stop', 'Visit date: 2026-10-10\nLisinopril was discontinued on 2026-10-10.')
    state = client.get(f"/api/patients/{morgan['id']}/facts/resolutions", params={'fact_type':'medication','entity':'lisinopril'}).json()['resolutions'][0]
    assert state['status'] == 'consistent'
    assert state['values']['resolved_state'] == 'discontinued'
    assert state['values']['as_of'] == '2026-10-10'
    thread = reopen_thread(client, morgan['id'])
    answer = ask(client, morgan['id'], thread['id'], 'What medications are documented?').json()['answer'].casefold()
    assert 'discontinued' in answer


def test_discontinued_then_restarted_resolves_current_without_reusing_old_dose_as_fresh(workspace):
    client, app, model, morgan, *_ = workspace
    add_note(client, morgan['id'], 'Lifecycle',
             'Lisinopril was discontinued on 2026-10-10.\nLisinopril was restarted on 2026-10-20.')
    state = medication_state_resolution(app.state.vault.db, morgan['id'], 'lisinopril')
    assert state is not None
    assert state.status == 'consistent'
    assert state.values['resolved_state'] == 'current'
    assert state.values['as_of'] == '2026-10-20'
    preferred = facts_for_patient(app.state.vault.db, morgan['id'], fact_type='medication_event', entity='lisinopril')
    assert any(f.event_type == 'restarted' and f.event_date == '2026-10-20' for f in preferred)
    thread = reopen_thread(client, morgan['id'])
    answer = ask(client, morgan['id'], thread['id'], 'What medications are documented?').json()['answer'].casefold()
    assert 'lisinopril (restarted on 2026-10-20; later current dose not documented)' in answer
    assert 'lisinopril 10 mg by mouth once daily' not in answer



def test_dated_restart_does_not_reuse_undated_current_dose(workspace):
    client, app, model, morgan, *_ = workspace
    add_note(client, morgan['id'], 'Undated amlodipine list',
             'Current medications: amlodipine 5 mg daily.')
    add_note(client, morgan['id'], 'Amlodipine restart',
             'Amlodipine was restarted on 2026-10-20.')
    row = client.get(
        f"/api/patients/{morgan['id']}/facts/resolutions",
        params={'fact_type':'medication','entity':'amlodipine'},
    ).json()['resolutions'][0]
    assert row['values']['resolved_state'] == 'current'
    thread = reopen_thread(client, morgan['id'])
    result = ask(client, morgan['id'], thread['id'], 'What medications are documented?').json()
    answer = result['answer'].casefold()
    assert 'amlodipine (restarted on 2026-10-20; later current dose not documented)' in answer
    assert 'amlodipine 5 mg daily' not in answer

def test_undated_current_and_discontinued_assertions_remain_conflict(workspace):
    client, app, model, morgan, *_ = workspace
    # A source with no record/visit date must not use DB update time as clinical chronology.
    add_note(client, morgan['id'], 'Undated contradiction',
             'Current medications: amlodipine 5 mg daily.\nDiscontinued medications: amlodipine 5 mg daily.')
    row = client.get(f"/api/patients/{morgan['id']}/facts/resolutions", params={'fact_type':'medication','entity':'amlodipine'}).json()['resolutions'][0]
    assert row['status'] == 'conflict'
    assert row['values']['resolved_state'] == 'conflict'


def test_measurement_same_date_conflict_is_visible(workspace):
    client, app, model, morgan, *_ = workspace
    add_note(client, morgan['id'], 'Lab A', 'Visit date: 2026-10-03\nCreatinine was 1.0 mg/dL on 2026-10-03.')
    add_note(client, morgan['id'], 'Lab B', 'Visit date: 2026-10-03\nCreatinine was 1.4 mg/dL on 2026-10-03.')
    rows = resolutions(client, morgan['id'], fact_type='measurement', entity='creatinine')
    row = next(item for item in rows if item['values']['effective_date'] == '2026-10-03')
    assert row['status'] == 'conflict'
    assert row['values']['values'] == ['1.0 mg/dL','1.4 mg/dL']
    thread = reopen_thread(client, morgan['id'])
    result = ask(client, morgan['id'], thread['id'], 'What are the latest labs and vitals?').json()
    assert 'conflict' in result['answer'].casefold()
    assert '1.0 mg/dL' in result['answer'] and '1.4 mg/dL' in result['answer']


def test_measurement_correction_supersedes_old_value(workspace):
    client, app, model, morgan, *_ = workspace
    add_note(client, morgan['id'], 'Lab original', 'Creatinine was 1.4 mg/dL on 2026-10-03.')
    add_note(client, morgan['id'], 'Lab correction',
             'Correction: creatinine was 1.0 mg/dL on 2026-10-03; prior value 1.4 mg/dL was incorrect.')
    rows = resolutions(client, morgan['id'], fact_type='measurement', entity='creatinine')
    row = next(item for item in rows if item['values']['effective_date'] == '2026-10-03')
    assert row['status'] in {'single','consistent'}
    assert row['values']['values'] == ['1.0 mg/dL']
    assert row['superseded_fact_ids']


def test_measurements_on_different_dates_form_time_series_not_conflict(workspace):
    client, app, model, morgan, *_ = workspace
    add_note(client, morgan['id'], 'Lab old', 'Creatinine was 1.4 mg/dL on 2026-10-03.')
    add_note(client, morgan['id'], 'Lab new', 'Creatinine was 1.0 mg/dL on 2026-10-04.')
    rows = resolutions(client, morgan['id'], fact_type='measurement', entity='creatinine')
    dated = {item['values']['effective_date']: item for item in rows if item['values']['effective_date'] in {'2026-10-03','2026-10-04'}}
    assert dated['2026-10-03']['status'] != 'conflict'
    assert dated['2026-10-04']['status'] != 'conflict'
    thread = reopen_thread(client, morgan['id'])
    result = ask(client, morgan['id'], thread['id'], 'What are the latest labs and vitals?').json()
    assert '1.0 mg/dL' in result['answer']
    assert '1.4 mg/dL' not in result['answer']


def test_source_edit_and_exclusion_recompute_conflict_resolution(workspace):
    client, app, model, morgan, *_ = workspace
    first = add_note(client, morgan['id'], 'Prescription A', 'Lisinopril was prescribed on 2026-09-01.')
    second = add_note(client, morgan['id'], 'Prescription B', 'Lisinopril was prescribed on 2026-09-03.')
    row = resolutions(client, morgan['id'], fact_type='medication_event', entity='lisinopril', event_type='prescribed')[0]
    assert row['status'] == 'conflict'
    excluded = client.post(f"/api/notes/{second['id']}/context", json={
        'enabled':False,'expected_version':1,'reason':'Resolve synthetic conflict by excluding source',
    })
    assert excluded.status_code == 200
    row = resolutions(client, morgan['id'], fact_type='medication_event', entity='lisinopril', event_type='prescribed')[0]
    assert row['status'] in {'single','consistent'}
    assert row['values']['event_dates'] == ['2026-09-01']
    updated = client.patch(f"/api/notes/{first['id']}", json={
        'title':'Prescription A','body':'Lisinopril was prescribed on 2026-09-02.',
        'expected_version':1,'reason':'Synthetic date correction',
    })
    assert updated.status_code == 200
    row = resolutions(client, morgan['id'], fact_type='medication_event', entity='lisinopril', event_type='prescribed')[0]
    assert row['values']['event_dates'] == ['2026-09-02']


def test_resolution_endpoint_is_patient_scoped_and_revocation_safe(workspace):
    client, app, model, morgan, jordan, *_ = workspace
    add_note(client, jordan['id'], 'Jordan conflict', 'Lisinopril was prescribed on 2025-01-01.')
    user, headers = create_user(client, 'doctor', 'resolution-doctor')
    assert client.get(f"/api/patients/{morgan['id']}/facts/resolutions", headers=headers).status_code == 404
    grant(client, morgan['id'], user['id'])
    allowed = client.get(f"/api/patients/{morgan['id']}/facts/resolutions", headers=headers)
    assert allowed.status_code == 200
    payload = json.dumps(allowed.json())
    assert '2025-01-01' not in payload
    assert client.post(f"/api/patients/{morgan['id']}/access", json={
        'user_id':user['id'],'enabled':False,'reason':'End resolution test grant',
    }).status_code == 200
    assert client.get(f"/api/patients/{morgan['id']}/facts/resolutions", headers=headers).status_code == 404


def test_semantic_resolution_data_remains_encrypted_at_rest(workspace):
    client, app, model, morgan, jordan, directory = workspace
    quote = 'Lisinopril was first prescribed on 2026-09-01.'
    add_note(client, morgan['id'], 'Encrypted semantics', quote)
    data = client.get(f"/api/patients/{morgan['id']}/facts/resolutions", params={
        'fact_type':'medication_event','entity':'lisinopril','event_type':'prescribed',
    }).json()
    assert data['resolutions']
    assert client.post('/api/lock').status_code == 200
    raw = (directory/'workspace.vault').read_bytes()
    assert quote.encode() not in raw
    assert b'medication_event|lisinopril|prescribed|original' not in raw
    assert b'2026-09-01' not in raw


def test_multiple_renewals_are_event_series_not_conflict(workspace):
    client, app, model, morgan, *_ = workspace
    add_note(client, morgan['id'], 'Renewal history',
             'Lisinopril was renewed on 2026-09-15.\nLisinopril was renewed on 2026-10-15.')
    rows = resolutions(client, morgan['id'], fact_type='medication_event', entity='lisinopril', event_type='renewed')
    assert len(rows) == 2
    assert {row['status'] for row in rows} <= {'single','consistent'}
    assert {tuple(row['values']['event_dates']) for row in rows} == {('2026-09-15',),('2026-10-15',)}
    thread = reopen_thread(client, morgan['id'])
    result = ask(client, morgan['id'], thread['id'], 'When was lisinopril renewed?').json()
    assert result['workflow_search']['status'] == 'documented_event_series'
    assert '2026-09-15' in result['answer'] and '2026-10-15' in result['answer']
    assert 'separate documented events' in result['answer'].casefold()


def test_same_order_id_with_conflicting_prescription_dates_is_conflict(workspace):
    client, app, model, morgan, *_ = workspace
    add_note(client, morgan['id'], 'Order A', 'Lisinopril prescription ID RX-77 was prescribed on 2026-09-01.')
    add_note(client, morgan['id'], 'Order B', 'Lisinopril prescription ID RX-77 was prescribed on 2026-09-03.')
    rows = resolutions(client, morgan['id'], fact_type='medication_event', entity='lisinopril', event_type='prescribed')
    matching = [row for row in rows if row['values'].get('order_id') == 'rx-77']
    assert len(matching) == 1
    assert matching[0]['status'] == 'conflict'
    assert matching[0]['values']['event_dates'] == ['2026-09-01','2026-09-03']


def test_corrupted_fact_under_resolution_is_rejected(workspace):
    client, app, model, morgan, *_ = workspace
    note = add_note(client, morgan['id'], 'Resolution corruption', 'Lisinopril was prescribed on 2026-09-11.')
    assert resolutions(client, morgan['id'], fact_type='medication_event', entity='lisinopril', event_type='prescribed')
    with app.state.vault.transaction() as db:
        db.execute("UPDATE clinical_facts SET quote='forged resolution quote' WHERE source_id=? AND fact_type='medication_event'", (note['id'],))
    response = client.get(f"/api/patients/{morgan['id']}/facts/resolutions", params={
        'fact_type':'medication_event','entity':'lisinopril','event_type':'prescribed',
    })
    assert response.status_code == 409
    assert 'provenance' in response.text.casefold()
