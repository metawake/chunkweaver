"""Synthetic labeled clinical notes for training and evaluation.

Each note is a list of (section_label, text) tuples. Section labels follow
standard clinical note sections. These are synthetic examples — no real
patient data — designed to cover common formatting patterns:

1. Clean headers (HPI:, ASSESSMENT:)
2. Implicit transitions (no header, topic changes mid-paragraph)
3. Abbreviated headers (PE, ROS, PMH)
4. Run-on dictated style
"""

from __future__ import annotations

LABELED_NOTES: list[list[tuple[str, str]]] = [
    # Note 1: Clean headers — baseline case
    [
        ("HPI", "HISTORY OF PRESENT ILLNESS:\n65 year old male presenting with acute onset chest pain radiating to left arm. Pain started 2 hours ago while at rest. Rates 8/10. Denies shortness of breath or diaphoresis. No prior episodes. Has not taken any medications for the pain."),
        ("PMH", "PAST MEDICAL HISTORY:\nHypertension diagnosed 2014, type 2 diabetes mellitus diagnosed 2018, hyperlipidemia. No prior cardiac history. No surgical history."),
        ("MEDS", "MEDICATIONS:\nMetformin 500mg BID, lisinopril 10mg daily, atorvastatin 40mg QHS, aspirin 81mg daily."),
        ("ALLERGIES", "ALLERGIES:\nNo known drug allergies."),
        ("FH", "FAMILY HISTORY:\nFather had myocardial infarction at age 58. Mother with hypertension. No family history of sudden cardiac death."),
        ("ROS", "REVIEW OF SYSTEMS:\nCardiovascular: chest pain as above. Respiratory: no cough, no dyspnea. GI: no nausea or vomiting. Neuro: no dizziness or syncope. All other systems negative."),
        ("PE", "PHYSICAL EXAMINATION:\nVitals: BP 168/94, HR 102, RR 18, O2 sat 98% on RA, Temp 98.6F.\nGeneral: Alert, oriented, appears uncomfortable.\nCardiac: Regular rate and rhythm, no murmurs, rubs or gallops.\nLungs: Clear to auscultation bilaterally.\nAbdomen: Soft, non-tender.\nExtremities: No edema, pulses 2+ bilaterally."),
        ("LABS", "LABORATORY DATA:\nTroponin I pending. CBC within normal limits. BMP notable for glucose 187. BNP 45."),
        ("IMAGING", "ECG shows ST elevation in leads II, III, aVF consistent with acute inferior STEMI. No prior ECG for comparison."),
        ("ASSESSMENT", "ASSESSMENT:\nAcute inferior ST-elevation myocardial infarction in a 65 year old male with cardiac risk factors including HTN, DM2, hyperlipidemia, and family history."),
        ("PLAN", "PLAN:\n1. Emergent cardiac catheterization lab activation\n2. Aspirin 325mg loading dose administered\n3. Heparin bolus and drip per protocol\n4. Clopidogrel 600mg loading dose\n5. Serial troponins q6h\n6. Admit to CCU post-procedure\n7. Cardiology consulted, Dr. Smith responding"),
    ],

    # Note 2: Implicit transitions — no headers, dictated style
    [
        ("HPI", "The patient is a 42 year old female who presents to the emergency department with a three day history of worsening abdominal pain localized to the right lower quadrant. She reports that the pain began as periumbilical discomfort and has migrated. She has had two episodes of emesis and reports anorexia for the past day. Last bowel movement was yesterday, normal consistency."),
        ("PMH", "Her past medical history is significant for endometriosis diagnosed in 2019 and a prior appendectomy is denied. She had a cesarean section in 2016 without complications."),
        ("MEDS", "She takes oral contraceptive pills and ibuprofen as needed for menstrual cramps. She denies any other medications or supplements."),
        ("ALLERGIES", "She reports an allergy to sulfa drugs which causes a rash."),
        ("ROS", "She denies fever, chills, diarrhea, constipation, urinary symptoms, or vaginal discharge. She denies any possibility of pregnancy. Last menstrual period was two weeks ago."),
        ("PE", "On examination she is afebrile with temperature 99.1, heart rate 94, blood pressure 128/78, respiratory rate 16, oxygen saturation 99% on room air. She appears uncomfortable and is lying still on the stretcher. Abdomen is soft with focal tenderness in the right lower quadrant. There is positive McBurney point tenderness. Rovsing sign is positive. No rebound but guarding is present. Bowel sounds are hypoactive."),
        ("LABS", "White blood cell count is 14.2 with left shift. Hemoglobin 13.1, platelets 234. Comprehensive metabolic panel is unremarkable. Lipase normal. Urinalysis negative. Urine pregnancy test negative."),
        ("IMAGING", "CT abdomen and pelvis with IV contrast demonstrates a dilated appendix measuring 12mm in diameter with surrounding fat stranding and a small amount of free fluid in the pelvis. No abscess identified. Findings consistent with acute uncomplicated appendicitis."),
        ("ASSESSMENT", "This is a 42 year old female with clinical and radiographic findings consistent with acute appendicitis."),
        ("PLAN", "Surgery has been consulted and Dr. Johnson will perform laparoscopic appendectomy this evening. Patient is NPO. IV fluids running. Pain managed with morphine 4mg IV. Preoperative antibiotics with ceftriaxone and metronidazole administered. Consent obtained."),
    ],

    # Note 3: Abbreviated headers, terse style
    [
        ("HPI", "CC: SOB x 2 days\nHPI: 78M w/ hx CHF (EF 35%), COPD on home O2, presents w/ worsening dyspnea. Unable to lie flat x 2 nights. 3-pillow orthopnea. +PND. Notes 5lb weight gain over past week. Increased LE edema. Denies chest pain, fever, cough productive of sputum. Has been compliant w/ meds but reports dietary indiscretion over holidays."),
        ("PMH", "PMH: CHF (ischemic, EF 35% on last echo 3mo ago), COPD (GOLD stage III), CAD s/p CABG x3 2018, AFib on warfarin, CKD stage 3, HTN, DM2."),
        ("MEDS", "Meds: furosemide 40mg BID, carvedilol 25mg BID, lisinopril 20mg daily, warfarin 5mg daily, metformin 1000mg BID, albuterol PRN, tiotropium daily, home O2 2L NC."),
        ("ALLERGIES", "Allergies: PCN (anaphylaxis)"),
        ("PE", "PE: T 98.4 HR 88 irreg BP 142/88 RR 24 O2 89% on 2L NC\nGen: elderly male, moderate respiratory distress, speaking in short sentences\nNeck: JVD to angle of jaw\nCV: irreg irreg, S3 gallop, no murmur\nLungs: bibasilar crackles to mid-lung fields bilaterally, no wheezing\nAbd: soft, NT, hepatomegaly 3cm below costal margin\nExt: 3+ pitting edema bilateral LE to knees"),
        ("LABS", "Labs: BNP 2840 (baseline 400), Cr 1.8 (baseline 1.4), K 5.1, Na 132, INR 2.3. Trop negative x1. CBC wnl."),
        ("IMAGING", "CXR: bilateral pleural effusions R>L, pulmonary vascular congestion, cardiomegaly."),
        ("ASSESSMENT", "Assessment: Acute on chronic systolic heart failure exacerbation, likely precipitated by dietary indiscretion. COPD stable. AKI likely prerenal vs cardiorenal."),
        ("PLAN", "Plan:\n- IV furosemide 80mg bolus then 40mg q8h, strict I/O, daily weights\n- Fluid restrict 1.5L/day, low sodium diet\n- Continue home carvedilol, hold lisinopril given AKI\n- Trend Cr, lytes BID\n- Continue warfarin, recheck INR in AM\n- O2 titrate to >92%, respiratory therapy consult\n- Echo in AM to reassess EF\n- If no improvement consider inotropic support"),
    ],

    # Note 4: Psychiatric note — different section structure
    [
        ("HPI", "PSYCHIATRIC EVALUATION\nReason for referral: 28 year old female referred by PCP for evaluation of depressive symptoms.\n\nPresenting complaint: Patient reports persistent low mood for approximately 4 months. She describes feeling sad most of the day nearly every day. She has lost interest in activities she previously enjoyed including running and socializing with friends. Sleep is disrupted with early morning awakening at 4am unable to return to sleep. Appetite is decreased with unintentional 12 pound weight loss. Concentration is impaired affecting her work performance. She denies suicidal ideation, plan, or intent. No history of suicide attempts. Denies homicidal ideation."),
        ("PMH", "Psychiatric history: No prior psychiatric treatment. No prior psychiatric hospitalizations. No history of self-harm.\n\nMedical history: Hypothyroidism on levothyroxine. Iron deficiency anemia. No other chronic conditions."),
        ("MEDS", "Current medications: Levothyroxine 75mcg daily, ferrous sulfate 325mg daily, multivitamin."),
        ("FH", "Family psychiatric history: Mother with history of depression treated with sertraline. Maternal grandmother with bipolar disorder. Father with alcohol use disorder in remission. No family history of completed suicide."),
        ("ROS", "Social history: Lives alone in apartment. Works as graphic designer remotely. Single, no children. Drinks 1-2 glasses of wine per week, denies binge drinking. Denies tobacco, marijuana, or other substance use. Has supportive friend group but has been isolating. No legal history."),
        ("PE", "Mental status examination:\nAppearance: Well-groomed, age-appropriate dress, poor eye contact.\nBehavior: Psychomotor slowing noted. Cooperative with interview.\nSpeech: Soft, slow rate, normal tone.\nMood: \"Empty.\"\nAffect: Flat, constricted range, congruent with stated mood.\nThought process: Linear, goal-directed, no loosening of associations.\nThought content: Negative for SI/HI, no delusions, no obsessions.\nPerceptions: Denies hallucinations, no evidence of internal stimulation.\nCognition: Alert and oriented x4. Concentration mildly impaired on serial 7s.\nInsight: Fair. Judgment: Fair."),
        ("ASSESSMENT", "Assessment: Major depressive disorder, single episode, moderate severity. PHQ-9 score 17 (moderately severe). Rule out hypothyroid-related mood symptoms — last TSH normal 2 months ago per PCP records."),
        ("PLAN", "Plan:\n1. Start sertraline 50mg daily, titrate to 100mg at 2 weeks if tolerated\n2. Discuss side effects: GI upset, sexual dysfunction, activation in first week\n3. Safety plan reviewed and documented\n4. Refer to cognitive behavioral therapy, weekly sessions\n5. Follow up in 2 weeks for medication check\n6. Recheck TSH, CBC\n7. Patient to call crisis line if SI develops\n8. Return precautions discussed"),
    ],

    # Note 5: Run-on dictated note — worst case for parsing
    [
        ("HPI", "This is a 55 year old gentleman with a history of poorly controlled type 2 diabetes and peripheral neuropathy who comes in today because he noticed a wound on the bottom of his right foot about a week ago that is not healing. He says he did not feel it happen and only noticed it when he saw blood on his sock. The wound has been getting larger and now has some drainage that he describes as yellowish. He has been soaking it in warm water. He denies fever or chills but his wife thinks the foot looks red and swollen."),
        ("PMH", "His medical history includes type 2 diabetes for about fifteen years, his last A1c was 9.2 about three months ago, he has diabetic neuropathy in both feet, he has hypertension, he had a right knee replacement two years ago, and he has chronic low back pain. He is a former smoker who quit ten years ago after a thirty pack year history."),
        ("MEDS", "He takes metformin 1000 twice a day, glipizide 10 in the morning, insulin glargine 30 units at bedtime, lisinopril 40 daily, gabapentin 300 three times a day for his neuropathy, and he takes ibuprofen as needed for his back."),
        ("PE", "On exam today his vitals are stable, temperature 99.8 orally, blood pressure 148/92, heart rate 78, his BMI is 34. Looking at his right foot there is a 2 by 1.5 centimeter ulcer on the plantar surface of the first metatarsal head. The wound base shows granulation tissue with some areas of fibrinous exudate. There is surrounding erythema extending about 2 centimeters from the wound margin. Mild warmth and edema of the forefoot. No crepitus. Sensation is diminished to monofilament testing in both feet. Dorsalis pedis pulse is palpable but diminished on the right. Left foot exam is unremarkable. His shoes show significant wear pattern suggesting abnormal gait mechanics."),
        ("LABS", "His glucose in clinic today is 242. We will get an A1c, CBC with differential, comprehensive metabolic panel, ESR and CRP. A wound culture was obtained. His last ABI was 0.72 on the right suggesting mild peripheral arterial disease."),
        ("ASSESSMENT", "So in summary we have a diabetic foot ulcer Wagner grade 2 on the right plantar surface with surrounding cellulitis in the setting of poorly controlled diabetes, peripheral neuropathy, and mild peripheral arterial disease. There is concern for possible early osteomyelitis given the depth and location of the wound."),
        ("PLAN", "I am going to start him on amoxicillin-clavulanate 875 twice daily empirically pending wound culture results. He needs to stay completely off the right foot, I am ordering a surgical shoe and crutches. I want to get an MRI of the right foot to evaluate for osteomyelitis. I am referring him to wound care clinic for debridement and ongoing management. I am also referring to endocrinology because his diabetes control needs significant improvement. I am stopping the ibuprofen given the wound and renal considerations and switching him to acetaminophen for his back. He needs to follow up with me in one week or sooner if the redness spreads or he develops fever. I showed him and his wife the signs of worsening infection to watch for."),
    ],

    # Note 6: ED note — rapid-fire sections
    [
        ("HPI", "32F brought in by EMS after witnessed seizure at workplace. Coworkers report generalized tonic-clonic activity lasting approximately 3 minutes followed by postictal confusion. No prior seizure history per coworkers. Patient fell during seizure, struck right side of head on desk. No tongue biting noted by witnesses. Incontinent of urine. On arrival patient is confused but following simple commands."),
        ("PMH", "Per medical records in system: migraine headaches, anxiety disorder on escitalopram. No seizure history. No head trauma history. No neurosurgical history."),
        ("MEDS", "Escitalopram 10mg daily per pharmacy records. No other medications."),
        ("PE", "VS: T 98.9 HR 112 BP 156/94 RR 20 O2 97% RA GCS 13 (E3V4M6)\nHEENT: 4cm laceration right temporal region, actively bleeding, no step-off, no hemotympanum\nNeuro: Pupils equal reactive 3mm, confused but moving all extremities symmetrically, no focal deficits appreciated, DTRs 2+ bilaterally\nCV: Tachycardic regular, no murmur\nLungs: Clear\nRemaining exam deferred given clinical status"),
        ("LABS", "Labs obtained: CBC, CMP, magnesium, calcium, phosphorus, glucose 94, ammonia, urine drug screen, urine pregnancy negative, prolactin level (drawn within 20 min of seizure). VBG 7.31/38/adequate bicarb. Lactate 3.2."),
        ("IMAGING", "CT head without contrast: No acute intracranial hemorrhage. No fracture. No midline shift."),
        ("ASSESSMENT", "New onset seizure in a 32 year old female, now in postictal state. Most common etiologies to consider: idiopathic/genetic epilepsy, drug/alcohol related, structural, metabolic. CT head reassuring. Will need further workup including MRI and EEG as outpatient."),
        ("PLAN", "Laceration repaired with 5 staples. Seizure precautions. Lorazepam 2mg IV available for recurrent seizure. Neurology consulted — recommend MRI brain with contrast as outpatient, outpatient EEG, start levetiracetam 500mg BID. No driving until cleared by neurology. Admit to observation for monitoring. Social work consulted to assist with work note and transportation."),
    ],

    # Note 7: Mixed formatting — some headers, some implicit
    [
        ("HPI", "CHIEF COMPLAINT: Back pain\n\nMr. Rodriguez is a 48 year old construction worker who presents with 5 days of progressive low back pain after lifting heavy materials at work. The pain is in the lumbar region, non-radiating, worse with bending and lifting, improved with rest and ibuprofen. He rates it 6/10 currently. No numbness, tingling, or weakness in the legs. No bowel or bladder dysfunction. No saddle anesthesia."),
        ("PMH", "He has had prior episodes of low back pain, the last one about a year ago that resolved with physical therapy. He has a history of GERD and takes omeprazole."),
        ("PE", "EXAMINATION:\nMusculoskeletal: tenderness over L4-L5 paraspinal muscles bilaterally. No midline tenderness. Range of motion limited in flexion by pain. Extension and lateral bending mildly limited. Negative straight leg raise bilaterally. Motor strength 5/5 in bilateral lower extremities. Sensation intact. Gait normal."),
        ("ASSESSMENT", "Acute mechanical low back pain without red flag features. Low suspicion for radiculopathy, cauda equina, or fracture given mechanism, exam findings, and no neurological deficits."),
        ("PLAN", "Conservative management recommended. Continue ibuprofen 600mg TID with food for 7 days. Added cyclobenzaprine 10mg QHS PRN for muscle spasm. Activity modification — avoid heavy lifting for 2 weeks. Referred to physical therapy. Work note provided for light duty for 2 weeks. Return if symptoms worsen, develop leg weakness, or bowel/bladder changes. No imaging indicated at this time per ACR appropriateness criteria."),
    ],

    # Note 8: Pediatric note
    [
        ("HPI", "PEDIATRIC VISIT\n4 year old male brought by mother for evaluation of persistent cough for 10 days. Cough is described as wet and productive, worse at night and with activity. Associated with rhinorrhea, initially clear now yellowish-green. Low grade fever to 100.4 three days ago that resolved. No ear pain. Decreased appetite but adequate fluid intake. No vomiting or diarrhea. Older sibling had similar symptoms 2 weeks ago that resolved. Attends daycare. Up to date on vaccinations per chart review."),
        ("PMH", "Past medical history: Full term delivery, no NICU stay. No chronic medical conditions. No prior hospitalizations. History of 2 episodes of otitis media in past year. No history of wheezing or asthma."),
        ("MEDS", "No current medications. No known drug allergies."),
        ("PE", "Vitals: Weight 17.2kg (50th percentile), Temp 98.8, HR 98, RR 22, O2 98%\nGeneral: Active, playful, no distress\nENT: TMs gray and mobile bilaterally, no effusion. Nasal mucosa erythematous with mucopurulent discharge. Oropharynx with mild cobblestoning, no exudate. No cervical lymphadenopathy.\nLungs: Transmitted upper airway sounds, no wheezing, no crackles, good air entry bilaterally.\nRemaining exam normal."),
        ("ASSESSMENT", "Acute viral upper respiratory infection with persistent post-nasal drip causing cough. Duration and progression consistent with viral URI. No clinical evidence of bacterial sinusitis (symptoms less than 10 days without worsening pattern), pneumonia, or acute otitis media."),
        ("PLAN", "Supportive care. Honey for cough (1-2 tsp as needed, appropriate for age over 1). Nasal saline irrigation. Cool mist humidifier at night. Adequate fluids. Acetaminophen or ibuprofen for comfort if needed. No antibiotics indicated. Return if fever returns, symptoms worsen after initial improvement, cough persists beyond 3 weeks, or any difficulty breathing. Anticipatory guidance provided to mother regarding typical URI course."),
    ],
]


def get_labeled_lines() -> list[tuple[str, str, bool]]:
    """Convert notes to per-line labeled data.

    Returns (line_text, section_label, is_section_start) for each
    non-empty line across all notes.
    """
    result: list[tuple[str, str, bool]] = []
    for note in LABELED_NOTES:
        for section_label, section_text in note:
            lines = section_text.split("\n")
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped:
                    result.append((stripped, section_label, i == 0))
    return result


def get_notes_as_text_and_labels() -> list[tuple[str, list[tuple[int, str]]]]:
    """Return each note as (full_text, [(line_number, section_label), ...]).

    Only section-start lines are included in the labels list.
    """
    result = []
    for note in LABELED_NOTES:
        full_text = "\n\n".join(text for _, text in note)
        lines = full_text.split("\n")
        labels: list[tuple[int, str]] = []
        offset = 0
        for section_label, section_text in note:
            section_lines = section_text.split("\n")
            labels.append((offset, section_label))
            offset += len(section_lines)
            offset += 1  # for the blank line between sections
        result.append((full_text, labels))
    return result
