"""Synthetic-only integration tests. No Internet or model weights required."""
import io
import json
import sqlite3
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.intake import identity, extract, MAX_UPLOAD
from app.main import create_app
from app.model import LocalModel, ModelUnavailable
from app.vault import Vault, MAGIC

VAULT_PASS = "test-only-vault-passphrase-not-real"
USER_PASS = "test-only-user-password-not-real"

class FakeModel:
    base_url = "http://127.0.0.1:10080/v1"
    model = "synthetic-test-double"
    remote = False
    def __init__(self):
        self.calls = []
        self.fail = False
        self.on_answer = None
    def status(self): return {"available": True, "models": [self.model], "remote": False}
    def summarize(self, text):
        self.calls.append(("summary", text))
        if self.fail: raise ModelUnavailable("Offline")
        return "Synthetic draft summary."
    def transcribe(self, image):
        self.calls.append(("transcribe", len(image)))
        return "Patient: Morgan Avery\nDOB: 1988-04-12\nMRN: SYN-1001\nSynthetic image text."
    def answer(self, question, context):
        self.calls.append(("answer", question, context))
        if self.fail: raise ModelUnavailable("Offline")
        if self.on_answer: self.on_answer()
        return "Synthetic test response [N1]. Verify originals."
    def close(self): pass

@pytest.fixture
def workspace(tmp_path):
    model = FakeModel()
    app = create_app(tmp_path, model)
    with TestClient(app) as client:
        assert client.post('/api/setup', json={"username":"admin", "password":USER_PASS, "vault_password":VAULT_PASS}).status_code == 200
        auth = client.post('/api/login', json={"username":"admin", "password":USER_PASS}).json()
        client.headers['Authorization'] = 'Bearer '+auth['token']
        assert client.post('/api/seed').status_code == 201
        patients = client.get('/api/patients').json()
        patient = next(p for p in patients if p['mrn']=='SYN-1001')
        yield client, app, model, patient, tmp_path

def upload(c, text=None, model=False, filename='synthetic.txt'):
    text = text or "Patient: Morgan Avery\nDOB: 1988-04-12\nMRN: SYN-1001\nSYNTHETIC RECORD. Uploaded update."
    return c.post('/api/intake', content=text.encode(), headers={'X-Filename':filename,'X-Use-Model':str(model).lower()})

def approve(c, doc, patient):
    return c.post(f"/api/documents/{doc['id']}/review", json={
        'patient_id':patient['id'], 'text':doc['text'], 'summary':'Human-reviewed synthetic update.',
        'expected_version':doc['version'], 'reason':'Reviewed against synthetic original',
        'identity_confirmed':True, 'content_confirmed':True})

def user(c, role='staff'):
    created=c.post('/api/users',json={'username':role,'password':USER_PASS,'role':role})
    assert created.status_code==201
    login=c.post('/api/login',json={'username':role,'password':USER_PASS}).json()
    return created.json()['id'], {'Authorization':'Bearer '+login['token']}


def test_locked_requires_unlock_and_individual_login(workspace):
    c,app,m,p,d=workspace
    assert c.post('/api/lock').status_code==200
    assert c.get('/api/patients').status_code==423
    assert c.post('/api/unlock',json={'password':'wrong-secret'}).status_code==401
    assert c.post('/api/unlock',json={'password':VAULT_PASS}).status_code==200
    assert c.get('/api/patients').status_code==401
    token=c.post('/api/login',json={'username':'admin','password':USER_PASS}).json()['token']
    assert c.get('/api/patients',headers={'Authorization':'Bearer '+token}).status_code==200


def test_ciphertext_and_backup_do_not_contain_plaintext(workspace):
    c,app,m,p,d=workspace
    raw=(d/'workspace.vault').read_bytes()
    assert raw.startswith(MAGIC)
    for value in [b'SQLite format',b'Morgan Avery',USER_PASS.encode(),VAULT_PASS.encode(),b'outside-records']:
        assert value not in raw
    assert c.get('/api/backup').content.startswith(MAGIC)
    assert {f.name for f in d.iterdir()}=={'.process.lock','workspace.vault'}
    assert (d/'workspace.vault').stat().st_mode & 0o777 == 0o600


def test_full_identity_autoattaches_but_does_not_enable_context(workspace):
    c,app,m,p,d=workspace
    r=upload(c);assert r.status_code==201
    doc=r.json()['document'];assert doc['patient_id']==p['id']
    assert doc['state']=='review' and doc['context_enabled']==0
    assert doc['id'] not in [s['id'] for s in c.get(f"/api/patients/{p['id']}/context").json()['sources']]
    assert approve(c,doc,p).status_code==200
    assert doc['id'] in [s['id'] for s in c.get(f"/api/patients/{p['id']}/context").json()['sources']]

@pytest.mark.parametrize('text',[
    'Patient: Morgan Avery\nSynthetic name-only update.',
    'Patient: Morgan Avery\nDOB: 1988-04-13\nMRN: SYN-1001\nSynthetic DOB mismatch.',
    'Patient: Different Person\nDOB: 1988-04-12\nMRN: SYN-1001\nSynthetic name mismatch.',
    'Patient: Morgan Avery\nPatient: Someone Else\nDOB: 1988-04-12\nMRN: SYN-1001',
])
def test_unsafe_identity_remains_unassigned(workspace,text):
    c,app,m,p,d=workspace
    r=upload(c,text);assert r.status_code==201
    assert r.json()['document']['patient_id'] is None
    assert len(c.get('/api/patients').json())==3


def test_identical_upload_is_deduplicated(workspace):
    c,app,m,p,d=workspace
    first=upload(c).json();second=upload(c).json()
    assert not first['duplicate'] and second['duplicate']
    assert first['document']['id']==second['document']['id']


def test_staff_is_denied_unassigned_chart_and_clinical_edits(workspace):
    c,app,m,p,d=workspace
    uid,headers=user(c)
    assert c.get('/api/patients',headers=headers).json()==[]
    assert c.get(f"/api/patients/{p['id']}",headers=headers).status_code==404
    c.post(f"/api/patients/{p['id']}/access",json={'user_id':uid,'enabled':True,'reason':'Assigned to care team'})
    assert c.get(f"/api/patients/{p['id']}",headers=headers).status_code==200
    assert c.post(f"/api/patients/{p['id']}/notes",json={'title':'Synthetic','body':'Denied'},headers=headers).status_code==403
    assert c.get(f"/api/patients/{p['id']}/context",headers=headers).status_code==403
    assert c.get(f"/api/patients/{p['id']}/audit",headers=headers).status_code==200


def test_unassigned_upload_does_not_reveal_existing_patient(workspace):
    c,app,m,p,d=workspace
    uid,headers=user(c)
    r=c.post('/api/intake',content=b'Patient: Morgan Avery\nDOB: 1988-04-12\nMRN: SYN-1001',headers={**headers,'X-Filename':'x.txt'})
    assert r.status_code==201 and r.json()['document']['patient_id'] is None
    assert 'Morgan Avery' not in r.json()['routing']


def test_doctor_permissions_revoked_and_account_disabled(workspace):
    c,app,m,p,d=workspace
    uid,headers=user(c,'doctor')
    grant={'user_id':uid,'enabled':True,'reason':'Assigned to care team'}
    c.post(f"/api/patients/{p['id']}/access",json=grant)
    assert c.get(f"/api/patients/{p['id']}/context",headers=headers).status_code==200
    c.post(f"/api/patients/{p['id']}/access",json={**grant,'enabled':False})
    assert c.get(f"/api/patients/{p['id']}/context",headers=headers).status_code==404
    c.patch(f'/api/users/{uid}',json={'active':False,'reason':'Access removed by admin'})
    assert c.get('/api/patients',headers=headers).status_code==401


def test_document_exclusion_removes_future_context_but_keeps_original(workspace):
    c,app,m,p,d=workspace
    doc=approve(c,upload(c).json()['document'],p).json()
    r=c.post(f"/api/documents/{doc['id']}/context",json={'enabled':False,'expected_version':doc['version'],'reason':'Incorrect source for this chart'})
    assert r.status_code==200
    context=c.get(f"/api/patients/{p['id']}/context").json()
    assert doc['id'] not in [s['id'] for s in context['sources']]
    assert c.get(f"/api/documents/{doc['id']}/file").status_code==200
    assert c.delete(f"/api/documents/{doc['id']}").status_code==405
    c.post(f"/api/patients/{p['id']}/ask",json={'question':'What is documented?','expected_revision':context['revision']})
    assert 'Uploaded update' not in m.calls[-1][-1]


def test_note_versions_and_optimistic_conflicts(workspace):
    c,app,m,p,d=workspace
    note=c.post(f"/api/patients/{p['id']}/notes",json={'title':'Synthetic note','body':'Original statement'}).json()
    update={'title':'Corrected synthetic note','body':'Corrected statement','expected_version':1,'reason':'Original was incorrect'}
    assert c.patch(f"/api/notes/{note['id']}",json=update).status_code==200
    assert c.patch(f"/api/notes/{note['id']}",json=update).status_code==409
    versions=c.get(f"/api/notes/{note['id']}/versions").json()
    assert [v['version'] for v in versions]==[2,1]
    assert versions[1]['body']=='Original statement'
    c.post(f"/api/notes/{note['id']}/context",json={'enabled':False,'expected_version':2,'reason':'Exclude corrected test note'})
    assert note['id'] not in [s['id'] for s in c.get(f"/api/patients/{p['id']}/context").json()['sources']]


def test_empty_context_never_calls_model(workspace):
    c,app,m,p,d=workspace
    empty=c.post('/api/patients',json={'name':'Synthetic Empty','dob':'1990-01-01','mrn':'EMPTY-01'}).json()
    r=c.post(f"/api/patients/{empty['id']}/ask",json={'question':'Summarize records','expected_revision':empty['revision']})
    assert r.status_code==200 and r.json()['model_used'] is False
    assert m.calls==[]


def test_stale_revision_rejected_before_model(workspace):
    c,app,m,p,d=workspace
    c.post(f"/api/patients/{p['id']}/notes",json={'title':'New synthetic note','body':'New record'})
    r=c.post(f"/api/patients/{p['id']}/ask",json={'question':'Summarize records','expected_revision':p['revision']})
    assert r.status_code==409 and m.calls==[]


def test_chart_changed_during_inference_discards_answer(workspace):
    c,app,m,p,d=workspace
    def change():
        with app.state.vault.transaction() as db:
            db.execute('UPDATE patients SET revision=revision+1 WHERE id=?',(p['id'],))
    m.on_answer=change
    r=c.post(f"/api/patients/{p['id']}/ask",json={'question':'What is documented?','expected_revision':p['revision']})
    assert r.status_code==409 and 'Synthetic test response' not in r.text


def test_model_outage_is_honest_and_has_no_cloud_fallback(workspace):
    c,app,m,p,d=workspace;m.fail=True
    r=upload(c,model=True);assert r.status_code==201
    assert r.json()['document']['summary_kind']!='model_draft'
    current=c.get(f"/api/patients/{p['id']}").json()['patient']
    assert c.post(f"/api/patients/{p['id']}/ask",json={'question':'What is documented?','expected_revision':current['revision']}).status_code==503


def test_read_only_tools_validate_arguments_and_chart_access(workspace):
    c,app,m,p,d=workspace
    assert c.get('/api/tools').json()['tools'][0]['type']=='function'
    good={'name':'get_patient_context','arguments':{'patient_id':p['id']}}
    assert c.post('/api/tools/invoke',json=good).status_code==200
    assert c.post('/api/tools/invoke',json={**good,'arguments':{'patient_id':p['id'],'sql':'DROP TABLE patients'}}).status_code==422
    assert c.post('/api/tools/invoke',json={**good,'name':'send_email'}).status_code==422
    uid,headers=user(c,'doctor')
    assert c.post('/api/tools/invoke',json=good,headers=headers).status_code==404


def test_audit_chain_and_append_only_triggers(workspace):
    c,app,m,p,d=workspace
    assert c.get('/api/audit/verify').json()['valid']
    with pytest.raises(sqlite3.IntegrityError):
        with app.state.vault.transaction() as db: db.execute("UPDATE audit SET action='tampered' WHERE seq=1")
    with pytest.raises(sqlite3.IntegrityError):
        with app.state.vault.transaction() as db: db.execute('DELETE FROM audit WHERE seq=1')
    assert c.get('/api/audit/verify').json()['valid']


def test_originals_and_drafts_never_enter_context_before_review(workspace):
    c,app,m,p,d=workspace
    doc=upload(c,model=True).json()['document']
    assert doc['summary']=='Synthetic draft summary.'
    context=c.get(f"/api/patients/{p['id']}/context").json()
    assert doc['id'] not in [s['id'] for s in context['sources']]
    assert approve(c,doc,p).json()['summary_kind']=='human_reviewed'
    context=c.get(f"/api/patients/{p['id']}/context").json()
    assert 'Synthetic draft summary' not in json.dumps(context)


def test_input_types_sizes_and_security_headers(workspace):
    c,app,m,p,d=workspace
    assert upload(c,'<script>alert(1)</script>',filename='bad.html').status_code==422
    assert c.post('/api/intake',content=b'x'*(MAX_UPLOAD+1),headers={'X-Filename':'large.txt'}).status_code==413
    assert c.get('/api/status',headers={'Origin':'https://attacker.example'}).status_code==403
    assert c.get('/api/status',headers={'Host':'attacker.example'}).status_code==400
    r=c.get('/');assert r.status_code==200
    assert r.headers['Cache-Control']=='no-store' and "connect-src 'self'" in r.headers['Content-Security-Policy']
    bad=c.post('/api/login',json={'username':'admin','password':USER_PASS,'extra':VAULT_PASS})
    assert bad.status_code==422 and VAULT_PASS not in bad.text and USER_PASS not in bad.text


def test_image_transcription_never_auto_routes(workspace):
    c,app,m,p,d=workspace
    image=io.BytesIO();Image.new('RGB',(100,100),'white').save(image,format='PNG')
    r=c.post('/api/intake',content=image.getvalue(),headers={'X-Filename':'handwritten.png','X-Use-Model':'true'})
    assert r.status_code==201 and r.json()['document']['patient_id'] is None
    assert r.json()['document']['context_enabled']==0
    assert sum(call[0]=='transcribe' for call in m.calls)==1


def test_digital_pdf_text_layer_extraction():
    from pypdf import PdfWriter
    from pypdf.generic import NameObject, DictionaryObject, DecodedStreamObject
    writer=PdfWriter();page=writer.add_blank_page(width=400,height=300)
    font=DictionaryObject({NameObject('/Type'):NameObject('/Font'),NameObject('/Subtype'):NameObject('/Type1'),NameObject('/BaseFont'):NameObject('/Helvetica')})
    page[NameObject('/Resources')]=DictionaryObject({NameObject('/Font'):DictionaryObject({NameObject('/F1'):writer._add_object(font)})})
    stream=DecodedStreamObject();stream.set_data(b'BT /F1 12 Tf 20 200 Td (Patient: Morgan Avery) Tj 0 -20 Td (DOB: 1988-04-12) Tj 0 -20 Td (MRN: SYN-1001) Tj ET')
    page[NameObject('/Contents')]=writer._add_object(stream)
    out=io.BytesIO();writer.write(out)
    model=FakeModel();result=extract('synthetic.pdf',out.getvalue(),model,False)
    assert result.method=='pdf_text' and result.identity['mrn']=='SYN-1001'
    assert model.calls==[]


def test_corrupted_vault_fails_closed(tmp_path):
    v=Vault(tmp_path);v.create(VAULT_PASS)
    with v.transaction(): v.audit('test',None,'synthetic.event','test',{})
    v.close();raw=bytearray(v.path.read_bytes());raw[-1]^=1;v.path.write_bytes(raw)
    with pytest.raises(Exception):v.unlock(VAULT_PASS)
    assert not v.unlocked


def test_atomic_snapshot_write_failure_rolls_memory_back(tmp_path,monkeypatch):
    v=Vault(tmp_path);v.create(VAULT_PASS)
    with v.transaction():v.audit('test',None,'before','test',{})
    old=v.path.read_bytes()
    def fail():raise OSError('Synthetic disk failure')
    monkeypatch.setattr(v,'persist',fail)
    with pytest.raises(OSError):
        with v.transaction():v.audit('test',None,'should_rollback','test',{})
    assert v.path.read_bytes()==old
    assert v.db.execute("SELECT count(*) FROM audit WHERE action='should_rollback'").fetchone()[0]==0
    v.close()


def test_second_process_cannot_open_same_vault(tmp_path):
    first=Vault(tmp_path);second=Vault(tmp_path);first.acquire_process_lock()
    with pytest.raises(RuntimeError):second.acquire_process_lock()
    first.close();second.acquire_process_lock();second.close()


def test_remote_model_requires_explicit_https_allowlist(monkeypatch):
    monkeypatch.setenv('MODEL_BASE_URL','https://remote.example/v1')
    monkeypatch.delenv('MODEL_ALLOW_REMOTE',raising=False)
    with pytest.raises(ValueError):LocalModel()
    monkeypatch.setenv('MODEL_ALLOW_REMOTE','true');monkeypatch.setenv('MODEL_REMOTE_ORIGIN','https://different.example')
    with pytest.raises(ValueError):LocalModel()
    monkeypatch.setenv('MODEL_BASE_URL','http://remote.example/v1');monkeypatch.setenv('MODEL_REMOTE_ORIGIN','http://remote.example')
    with pytest.raises(ValueError):LocalModel()


def test_sdk_request_contract_when_sdk_installed(monkeypatch):
    # Real inference is deliberately not asserted. A mock HTTP transport verifies wire format.
    import app.model as module
    if module.OpenAI is None:pytest.skip('OpenAI SDK unavailable in offline build environment')
    import httpx
    requests=[]
    def handler(request):
        requests.append(request)
        return httpx.Response(200,json={'id':'synthetic','object':'chat.completion','created':0,'model':'smolvlm','choices':[{'index':0,'message':{'role':'assistant','content':'Synthetic answer'},'finish_reason':'stop'}]})
    m=LocalModel();m.client.close()
    m.client=module.OpenAI(api_key='local-development',base_url=m.base_url,http_client=httpx.Client(transport=httpx.MockTransport(handler),trust_env=False),max_retries=0)
    assert m.complete('instructions','synthetic input',b'png-bytes')=='Synthetic answer'
    request=requests[0];body=json.loads(request.content)
    assert str(request.url)=='http://127.0.0.1:10080/v1/chat/completions'
    assert body['messages'][1]['content'][1]['image_url']['url'].startswith('data:image/png;base64,')
    m.close()

@pytest.mark.parametrize('action,expected',[('lock',409),('logout',401)])
def test_lock_or_logout_during_inference_discards_answer(workspace,action,expected):
    c,app,m,p,d=workspace
    m.on_answer=lambda:c.post('/api/'+action)
    r=c.post(f"/api/patients/{p['id']}/ask",json={'question':'What is documented?','expected_revision':p['revision']})
    assert r.status_code==expected and 'Synthetic test response' not in r.text
