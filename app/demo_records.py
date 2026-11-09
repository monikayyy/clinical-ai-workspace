"""Detailed fictional records used for the local demonstration workspace."""
from __future__ import annotations

DEMO_RECORDS = [
    {
        "name": "Morgan Avery",
        "dob": "1988-04-12",
        "mrn": "SYN-1001",
        "title": "Primary care clinical summary",
        "legacy_titles": ["Follow-up documentation", "Records review"],
        "note": """DEMO DATA — FICTIONAL PATIENT
Active problems: essential hypertension and seasonal allergic rhinitis.
Current medications: lisinopril 10 mg by mouth once daily; cetirizine 10 mg by mouth as needed for allergies.
Medication allergy: penicillin caused a rash.
Most recent reviewed visit (2026-09-28): blood pressure 128/78 mmHg, pulse 72 beats/min, temperature 98.6 F. The patient denied chest pain, shortness of breath, dizziness, and medication side effects.
Documented plan: continue the existing medication list, keep a home blood-pressure log, and return in three months. An outside cardiology report was requested and remains pending.
No clinician-approved fever treatment plan is documented in the reviewed chart.""",
        "source": """DEMO DATA — FICTIONAL PATIENT
Patient: Morgan Avery
DOB: 1988-04-12
MRN: SYN-1001
Report date: 2026-10-01
Document type: outside records review

The reviewed outside summary lists essential hypertension and seasonal allergic rhinitis. Lisinopril 10 mg daily and cetirizine 10 mg as needed are on the medication list. Penicillin allergy with rash is documented. The most recent blood pressure was 128/78 mmHg. No interval cardiopulmonary symptoms were reported. The requested cardiology report has not yet been received. No fever treatment plan is included in this record.""",
        "summary": "Hypertension and allergic rhinitis are documented; medications and penicillin allergy were reconciled, and the cardiology report remains pending.",
        "filename": "outside-records-synthetic.txt",
        "legacy_filenames": ["outside-records-synthetic.txt"],
    },
    {
        "name": "Jordan Ellis",
        "dob": "1974-09-03",
        "mrn": "SYN-1002",
        "title": "Diabetes follow-up summary",
        "legacy_titles": ["Initial intake"],
        "note": """DEMO DATA — FICTIONAL PATIENT
Active problems: type 2 diabetes mellitus and hyperlipidemia.
Current medications: metformin 500 mg by mouth twice daily with meals; atorvastatin 20 mg by mouth nightly.
Medication allergies: none documented.
Laboratory results reviewed on 2026-09-20: hemoglobin A1c 7.2 percent, LDL cholesterol 96 mg/dL, creatinine 0.9 mg/dL.
The patient reported taking medications consistently and denied symptomatic hypoglycemia. A dilated retinal examination result is still outstanding.
Documented plan: continue the recorded regimen, repeat A1c in three months, and obtain the retinal examination report.""",
        "source": """DEMO DATA — FICTIONAL PATIENT
Patient: Jordan Ellis
DOB: 1974-09-03
MRN: SYN-1002
Report date: 2026-09-20
Document type: laboratory and medication review

Hemoglobin A1c was 7.2 percent, LDL cholesterol was 96 mg/dL, and creatinine was 0.9 mg/dL. Metformin 500 mg twice daily and atorvastatin 20 mg nightly are listed. No medication allergy is documented. The retinal examination report is pending.""",
        "summary": "Diabetes and lipid results were reviewed; medications are documented and the retinal examination report remains pending.",
        "filename": "diabetes-followup-demo.txt",
        "legacy_filenames": [],
    },
    {
        "name": "Taylor Quinn",
        "dob": "1996-02-21",
        "mrn": "SYN-1003",
        "title": "Asthma and allergy summary",
        "legacy_titles": ["Records review"],
        "note": """DEMO DATA — FICTIONAL PATIENT
Active problems: mild intermittent asthma and allergic rhinitis.
Current medications: albuterol HFA inhaler, two puffs as needed for wheezing; fluticasone nasal spray, one spray per nostril daily during allergy season.
Medication allergy: sulfonamide antibiotics caused hives.
Recent review (2026-09-18): no emergency visits, nighttime symptoms, or oral steroid use in the prior six months. Lungs were documented as clear, oxygen saturation 99 percent on room air.
Documented plan: continue the recorded rescue inhaler plan and arrange routine follow-up. Pulmonary-function testing has not yet been completed.""",
        "source": """DEMO DATA — FICTIONAL PATIENT
Patient: Taylor Quinn
DOB: 1996-02-21
MRN: SYN-1003
Report date: 2026-09-18
Document type: asthma follow-up

Mild intermittent asthma and allergic rhinitis are documented. Albuterol HFA as needed and seasonal fluticasone nasal spray are listed. Sulfonamide antibiotic allergy with hives is documented. No recent emergency visits, nighttime symptoms, or oral steroid use were reported. Pulmonary-function testing remains pending.""",
        "summary": "Asthma is stable by the reviewed history; rescue and allergy medications are documented, and pulmonary-function testing remains pending.",
        "filename": "asthma-followup-demo.txt",
        "legacy_filenames": [],
    },
]
