"""Persistent provenance fact-store regressions."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.fact_store import EXTRACTION_VERSION, facts_for_patient, validate_fact_provenance
from app.main import create_app
from app.vault import (
    ASSISTANT_MEMORY_DESCRIPTION,
    ASSISTANT_MEMORY_VERSION,
    FACT_SEMANTICS_DESCRIPTION,
    PROVENANCE_FACT_DESCRIPTION,
    PROVENANCE_FACT_VERSION,
    SCHEMA_VERSION,
    Vault,
    now,
)
from tests.test_assistant_threads import ADMIN_PASS, ThreadModel, VAULT_PASS, create_user, grant, workspace  # noqa: F401
from tests.test_medication_event_workflow import add_note, reopen_thread


def test_new_workspace_has_schema5_fact_tables_and_seeded_inventory(workspace):
    client, app, model, morgan, jordan, directory = workspace
    status = client.get('/api/status').json()
    assert status['schema_version'] == SCHEMA_VERSION == 5
    assert status['stage2_increment'] == 'provenance-conflict-semantics'
    assert status['workflow_revision'] == EXTRACTION_VERSION
    with app.state.vault.mutex:
        db = app.state.vault.db
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {'clinical_fact_sources', 'clinical_facts'} <= tables
        migration = db.execute('SELECT description FROM schema_migrations WHERE version=4').fetchone()[0]
        assert migration == PROVENANCE_FACT_DESCRIPTION
        semantic_migration = db.execute('SELECT description FROM schema_migrations WHERE version=5').fetchone()[0]
        assert semantic_migration == FACT_SEMANTICS_DESCRIPTION
        assert 'clinical_fact_resolutions' in tables
    data = client.get(f"/api/patients/{morgan['id']}/facts").json()
    assert data['inventory']['extraction_version'] == EXTRACTION_VERSION
    assert data['inventory']['fact_count'] >= 10
    assert {'medication','condition','allergy','pending_item','measurement','plan'} <= set(data['inventory']['by_type'])
    meds = [fact for fact in data['facts'] if fact['fact_type'] == 'medication']
    assert {fact['entity_label'].casefold() for fact in meds} >= {'lisinopril','cetirizine'}
    assert all(fact['source']['quote'] for fact in data['facts'])


def test_explicit_medication_events_are_persisted_with_exact_provenance(workspace):
    client, app, model, morgan, *_ = workspace
    note = add_note(client, morgan['id'], 'Medication event timeline',
                    'Lisinopril was prescribed on 2026-09-04.\nLisinopril was dispensed on 2026-09-06.')
    response = client.get(f"/api/patients/{morgan['id']}/facts?fact_type=medication_event&entity=lisinopril")
    assert response.status_code == 200
    facts = response.json()['facts']
    assert {(fact['event_type'], fact['event_date']) for fact in facts} == {
        ('prescribed','2026-09-04'), ('dispensed','2026-09-06')
    }
    for fact in facts:
        assert fact['source']['id'] == note['id']
        assert fact['source']['version'] == 1
        assert fact['source']['quote'] in {
            'Lisinopril was prescribed on 2026-09-04.',
            'Lisinopril was dispensed on 2026-09-06.',
        }
        assert fact['source']['end'] > fact['source']['start']
        assert len(fact['source']['sha256']) == 64
        assert fact['extraction_method'] == 'deterministic_rules'


def test_note_edit_replaces_old_fact_version_and_date(workspace):
    client, app, model, morgan, *_ = workspace
    note = add_note(client, morgan['id'], 'Prescription timeline', 'Lisinopril was prescribed on 2026-09-04.')
    before = client.get(f"/api/patients/{morgan['id']}/facts?fact_type=medication_event&entity=lisinopril&event_type=prescribed").json()['facts']
    assert any(fact['event_date'] == '2026-09-04' for fact in before)
    updated = client.patch(f"/api/notes/{note['id']}", json={
        'title': 'Prescription timeline',
        'body': 'Lisinopril was prescribed on 2026-09-05.',
        'expected_version': 1,
        'reason': 'Correct synthetic prescription date',
    })
    assert updated.status_code == 200, updated.text
    after = client.get(f"/api/patients/{morgan['id']}/facts?fact_type=medication_event&entity=lisinopril&event_type=prescribed").json()['facts']
    matching = [fact for fact in after if fact['source']['id'] == note['id']]
    assert len(matching) == 1
    assert matching[0]['event_date'] == '2026-09-05'
    assert matching[0]['source']['version'] == 2
    assert all(fact['event_date'] != '2026-09-04' for fact in matching)


def test_context_exclusion_removes_derived_facts_and_reinclude_rebuilds(workspace):
    client, app, model, morgan, *_ = workspace
    note = add_note(client, morgan['id'], 'Dispensing timeline', 'Lisinopril was dispensed on 2026-09-06.')
    assert any(f['source']['id'] == note['id'] for f in client.get(
        f"/api/patients/{morgan['id']}/facts?fact_type=medication_event&entity=lisinopril&event_type=dispensed"
    ).json()['facts'])
    excluded = client.post(f"/api/notes/{note['id']}/context", json={
        'enabled': False, 'expected_version': 1, 'reason': 'Synthetic exclusion test',
    })
    assert excluded.status_code == 200
    facts = client.get(f"/api/patients/{morgan['id']}/facts?fact_type=medication_event&entity=lisinopril&event_type=dispensed").json()['facts']
    assert all(fact['source']['id'] != note['id'] for fact in facts)
    included = client.post(f"/api/notes/{note['id']}/context", json={
        'enabled': True, 'expected_version': 2, 'reason': 'Synthetic reinclude test',
    })
    assert included.status_code == 200
    facts = client.get(f"/api/patients/{morgan['id']}/facts?fact_type=medication_event&entity=lisinopril&event_type=dispensed").json()['facts']
    restored = [fact for fact in facts if fact['source']['id'] == note['id']]
    assert len(restored) == 1 and restored[0]['source']['version'] == 3


def test_fact_api_enforces_patient_grants(workspace):
    client, app, model, morgan, *_ = workspace
    doctor, headers = create_user(client, 'doctor', 'fact-doctor')
    assert client.get(f"/api/patients/{morgan['id']}/facts", headers=headers).status_code == 404
    grant(client, morgan['id'], doctor['id'])
    assert client.get(f"/api/patients/{morgan['id']}/facts", headers=headers).status_code == 200
    revoke = client.post(f"/api/patients/{morgan['id']}/access", json={
        'user_id': doctor['id'], 'enabled': False, 'reason': 'End fact-store test grant',
    })
    assert revoke.status_code == 200
    assert client.get(f"/api/patients/{morgan['id']}/facts", headers=headers).status_code == 404


def test_direct_source_hash_drift_is_self_healed_before_fact_read(workspace):
    client, app, model, morgan, *_ = workspace
    note = add_note(client, morgan['id'], 'Hash drift', 'Lisinopril was prescribed on 2026-09-04.')
    initial = client.get(f"/api/patients/{morgan['id']}/facts?fact_type=medication_event&entity=lisinopril&event_type=prescribed").json()['facts']
    assert any(f['event_date'] == '2026-09-04' and f['source']['id'] == note['id'] for f in initial)
    # Simulate non-API corruption/drift without a version bump. Lazy sync must use the content hash.
    with app.state.vault.transaction() as db:
        db.execute("UPDATE notes SET body=? WHERE id=?", ('Lisinopril was prescribed on 2026-09-07.', note['id']))
    refreshed = client.get(f"/api/patients/{morgan['id']}/facts?fact_type=medication_event&entity=lisinopril&event_type=prescribed").json()['facts']
    fact = next(f for f in refreshed if f['source']['id'] == note['id'])
    assert fact['event_date'] == '2026-09-07'
    assert fact['source']['version'] == 1
    assert '2026-09-04' not in fact['source']['quote']


def test_corrupted_fact_row_is_rejected_not_returned(workspace):
    client, app, model, morgan, *_ = workspace
    note = add_note(client, morgan['id'], 'Fact corruption', 'Lisinopril was prescribed on 2026-09-04.')
    with app.state.vault.transaction() as db:
        db.execute("UPDATE clinical_facts SET quote='forged quote' WHERE source_kind='note' AND source_id=?", (note['id'],))
    response = client.get(f"/api/patients/{morgan['id']}/facts?fact_type=medication_event&entity=lisinopril")
    assert response.status_code == 409
    assert 'fact index no longer matches' in response.text.casefold()


def test_uncertain_or_negated_medication_event_is_not_indexed_as_fact(workspace):
    client, app, model, morgan, *_ = workspace
    add_note(client, morgan['id'], 'Unsafe event wording',
             'Lisinopril was not prescribed on 2026-09-04. Lisinopril may have been dispensed on 2026-09-06.')
    facts = client.get(f"/api/patients/{morgan['id']}/facts?fact_type=medication_event&entity=lisinopril").json()['facts']
    assert not any(f['event_date'] in {'2026-09-04','2026-09-06'} for f in facts)


def test_medication_event_workflow_reads_persisted_fact_index(workspace, monkeypatch):
    client, app, model, morgan, *_ = workspace
    add_note(client, morgan['id'], 'Indexed prescription', 'Lisinopril was prescribed on 2026-09-04.')
    # Ensure index is current, then make the old per-request parser fail. A successful query proves the persisted fact is used.
    assert client.get(f"/api/patients/{morgan['id']}/facts?fact_type=medication_event&entity=lisinopril").status_code == 200
    import app.medication_event_workflow as workflow
    def fail_parser(*args, **kwargs):
        raise AssertionError('per-request event parser should not run for an indexed explicit fact')
    monkeypatch.setattr(workflow, '_event_mentions', fail_parser)
    thread = reopen_thread(client, morgan['id'])
    response = client.post(f"/api/patients/{morgan['id']}/ask", json={
        'question': 'When was lisinopril prescribed?',
        'expected_revision': client.get(f"/api/patients/{morgan['id']}").json()['patient']['revision'],
        'thread_id': thread['id'], 'history': [],
    })
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['answer_origin'] == 'workflow_search'
    assert '2026-09-04' in data['answer']
    assert data['workflow_search']['coverage'] == 'provenance_fact_index_current_sources'


def test_fact_store_is_encrypted_at_rest(workspace):
    client, app, model, morgan, jordan, directory = workspace
    secret = 'Lisinopril was prescribed on 2042-12-24.'
    add_note(client, morgan['id'], 'Encrypted fact fixture', secret)
    response = client.get(f"/api/patients/{morgan['id']}/facts?fact_type=medication_event&entity=lisinopril")
    assert response.status_code == 200
    assert any('2042-12-24' in fact['source']['quote'] for fact in response.json()['facts'])
    raw = (directory / 'workspace.vault').read_bytes()
    assert secret.encode() not in raw
    assert b'2042-12-24' not in raw
    assert b'SQLite format 3' not in raw


def test_version3_vault_migrates_to_empty_fact_store_preserving_threads(tmp_path):
    vault = Vault(tmp_path)
    vault.create(VAULT_PASS)
    timestamp = now()
    with vault.transaction() as db:
        db.execute("INSERT INTO users(id,username,role,password_hash,created_at) VALUES('u','admin','admin','hash',?)", (timestamp,))
        db.execute("INSERT INTO patients(id,name,dob,mrn,created_at,updated_at) VALUES('p','Synthetic Patient','1980-01-01','SYN-PF',?,?)", (timestamp, timestamp))
        db.execute("INSERT INTO grants VALUES('p','u')")
        db.execute("""INSERT INTO assistant_threads(id,user_id,patient_id,session_id,scope,title,status,chart_revision,version,created_at,updated_at)
                      VALUES('t','u','p',NULL,'longitudinal','Saved thread','active',1,1,?,?)""", (timestamp,timestamp))
        db.execute("""INSERT INTO assistant_messages(id,thread_id,sequence,role,content,answer_origin,source_ids_json,chart_revision,created_at)
                      VALUES('m','t',1,'user','Saved question',NULL,'[]',1,?)""", (timestamp,))
        vault.audit('u','p','assistant.test','t',{'synthetic': True})
    with vault.transaction() as db:
        db.execute('DROP TABLE clinical_fact_resolutions')
        db.execute('DROP TABLE clinical_facts')
        db.execute('DROP TABLE clinical_fact_sources')
        db.execute('DELETE FROM schema_migrations WHERE version>=4')
        db.execute('PRAGMA user_version=3')
    vault.lock()

    migrated = Vault(tmp_path)
    migrated.unlock(VAULT_PASS)
    assert migrated.schema_version() == 5
    assert migrated.last_migrations == [
        f"Migrated vault to schema version {PROVENANCE_FACT_VERSION}: {PROVENANCE_FACT_DESCRIPTION}",
        f"Migrated vault to schema version 5: {FACT_SEMANTICS_DESCRIPTION}",
    ]
    assert migrated.db.execute("SELECT content FROM assistant_messages WHERE id='m'").fetchone()[0] == 'Saved question'
    assert migrated.db.execute('SELECT COUNT(*) FROM clinical_facts').fetchone()[0] == 0
    assert migrated.verify_audit()['valid']
    migrated.close()


def test_documented_medication_is_not_promoted_to_current_status(workspace):
    client, app, model, morgan, jordan, directory = workspace
    patient = client.post('/api/patients', json={
        'name': 'Historical Medication Synthetic', 'dob': '1980-01-01', 'mrn': 'SYN-HIST-MED',
    }).json()
    created = client.post(f"/api/patients/{patient['id']}/notes", json={
        'title': 'Historical medication mention',
        'body': 'Medication appearing in records: lisinopril 10 mg tablet.',
    })
    assert created.status_code == 201, created.text
    facts = client.get(f"/api/patients/{patient['id']}/facts", params={'fact_type':'medication'}).json()['facts']
    assert len(facts) == 1
    assert facts[0]['status'] == 'documented'
    detail = client.get(f"/api/patients/{patient['id']}").json()
    answer = client.post(f"/api/patients/{patient['id']}/ask", json={
        'question': 'What medications are documented?',
        'expected_revision': detail['patient']['revision'],
    }).json()
    assert answer['answer_origin'] == 'provenance_fact_store'
    assert 'does not establish that they are current medications' in answer['answer']
    assert 'The current medications are' not in answer['answer']
