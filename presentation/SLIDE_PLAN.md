# SinhaSpeech Midterm Slide Plan

This is the agreed slide order for the midterm presentation, built from the actual work on `main`, `Yohan_Observation`, and `Yohan_Finetune` (checked directly against the repo and commit history, not from memory). It tells each person exactly which existing template slide to duplicate, which placeholder gets which text, and what to do with each image or icon spot. Read the "How to use this" section first, then jump to your slides.

## How to use this

- The slide numbers below are final. If your section needs more room, add a sub-slide at the end of your range (for example `10a` after slide 10) instead of shifting the slides after yours.
- Every slide says "duplicate template slide N": find that slide in the shared file, copy it to the right spot in the new deck order, then edit the text and visuals in place. Don't build slides from scratch; the template's spacing and colors are already tuned.
- Slide 5 is the one slide the three of us should write together before anyone touches their own section. It sets the order (data, then model, then app) that the rest of the deck follows, so it needs to be agreed on first, not drafted solo and adjusted around later.
- Only slide 14 uses a real photo. Every other slide should keep the template's line-icon style or swap in an actual diagram or screenshot from the repo. No new stock photography.
- Fine-tuning has a working pipeline and is ready to run on GPU, but hasn't produced a trained model yet (`Yohan_Finetune` isn't merged and `eval_results.jsonl` only has a smoke-test entry). Say "ready to launch," not "trained" or "done," on slide 12.

## Who did what

**Yohan** built the data side: pulled in four raw sources (OpenSLR, YouTube, BizBrains, Linga), cleaned them (removed silent/noisy clips with voice activity detection, deduplicated across sources, which threw out about two-thirds of the collected set as copies of OpenSLR audio, stripped invisible Unicode characters, and manually reviewed flagged transcripts), checked the cleaned data still matched the original distribution using KL and Jensen-Shannon divergence, and assembled a final pool of 154,828 rows split 80/10/10 for training. On a separate branch not yet merged, Yohan also built the full GPU fine-tuning setup: both a full fine-tune and a LoRA fine-tune of Whisper-small, Weights & Biases logging, ranked multi-run evaluation, and a RunPod deployment guide.

**Imindu** built the application: the React frontend (lecture upload, spoken quizzes, transcript review and editing, teacher tools, accessibility settings) and the FastAPI/PostgreSQL backend (JWT login, a transcript database and API, an upload queue with a background worker, protected media streaming). Imindu then connected two real transcription models to that backend behind a config switch, replacing the placeholder that used to return canned text.

**Tharupahan** evaluated existing Sinhala speech-to-text models before committing to build one from scratch, built the fine-tuning and evaluation pipeline (training script, scoring script, shared library code), defined the exact handoff point where a trained model plugs into Imindu's backend, and read the project's requirements and design documents cover to cover to check the pipeline against them, catching several gaps early.

## Template layout reference

A quick index of the 13 existing slides, so everyone knows which one to duplicate for a given kind of content.

| # | Nickname in template | Structure | Best for |
|---|---|---|---|
| 1 | Title | Centered wordmark, small tag bottom right | Opening slide only |
| 2 | Project Overview | Eyebrow, title, three columns each with an icon, a heading, and 2-3 lines | Any three-part breakdown |
| 3 | Project Objective | Left text panel (a goal statement plus a bulleted list), right half full-height photo | One central point plus supporting detail |
| 4 | What We Found | Eyebrow, title, three numbered columns, then a curved divider into an italic "Key Insight" line | Findings that lead to one takeaway |
| 5 | Who We Designed For (split) | Two icon-and-text blocks side by side, sitting above a full-width photo strip | Two audiences or two contrasted things |
| 6 | Who We Designed For (chevron) | A four-segment arrow bar, each segment with a heading and a paragraph underneath | A four-step process, all steps equal weight |
| 7 | Methodology | A horizontal timeline with labels alternating above and below the line | Anything chronological, or steps that pair up |
| 8 | Project Summary | Left photo, right text panel with two stacked heading-plus-paragraph sections | A product or feature summary anchored by one hero image |
| 9 | Key Features | Left text panel with one long paragraph, right half photo | A single idea that needs room to breathe |
| 10 | Prototype | Top two-thirds full-bleed photo, bottom band with three short captions | Real screenshots, the only slide meant for a live photo |
| 11 | Challenges Faced | Two columns, each a heading plus a paragraph | Any side-by-side pair: two challenges, before/after, findings/fixes |
| 12 | What We Learned | A 2x2 grid of heading-plus-paragraph blocks | Any four-part breakdown: learnings, next steps, roles |
| 13 | Thank You | Centered closing text | Closing slide only |

## The deck, slide by slide

### Slide 1: Title
**Owner:** whoever opens the talk. **Duplicate:** template slide 1.
Keep as is: "SinhaSpeech" wordmark, "Group 12" tag in the bottom right. No edits needed.

### Slide 2: Project Overview
**Owner:** team. **Duplicate:** template slide 2.
- Eyebrow: `INTRODUCTION`
- Title: `Project Overview`
- Column 1 heading `Problem`, body: Sinhala-speaking students and teachers don't have an accessible way to turn lectures and spoken answers into text they can search, edit, or study from.
- Column 2 heading `Solution`, body: a web platform that records or accepts uploaded audio, transcribes it in Sinhala, and lets students and teachers review, edit, and quiz from the result.
- Column 3 heading `Why It Matters`, body: makes lecture content and spoken assessment usable for students who benefit from text, and gives teachers a faster way to review spoken answers.
- Icons: keep the existing three stick-figure doodles (the thinking figure, the reading figure, the lightbulb figure). They're generic enough to work here without a swap.

### Slide 3: Project Objectives
**Owner:** team. **Duplicate:** template slide 3.
- Eyebrow: `INTRODUCTION`
- Title: `Project Objectives`
- Main Goal: build a Sinhala speech-to-text platform accurate and fast enough for real classroom use, covering lecture transcription, study notes, and spoken quizzes.
- Supporting Objectives (bullets): an accurate Sinhala ASR model trained on a properly cleaned dataset; a backend that can queue, process, and store transcriptions reliably; a frontend that makes reviewing and correcting a transcript straightforward for both students and teachers.
- Photo panel (right half): replace the stock notebook photo. Best option is a screenshot of the transcript review screen in the frontend if one is ready by presentation time; otherwise leave the panel as a plain colored block rather than an unrelated photo.

### Slide 4: Who It's For
**Owner:** team. **Duplicate:** template slide 5.
- Eyebrow: `TARGET AUDIENCE`
- Title: `Who We Built This For`
- Block 1 heading `Students`, body: need lecture and self-study audio turned into text they can search, correct, and study from, plus a way to answer quizzes by speaking instead of typing.
- Block 2 heading `Teachers`, body: need to review and grade spoken answers quickly, and manage quizzes without listening to every recording end to end.
- Icons: swap in two different stick figures from the same doodle set: one reading/writing for students, one checking a list or grading for teachers.
- Bottom photo strip: drop it, or replace with a plain color band, since the current photo (a planner on a desk) has nothing to do with the product.

### Slide 5: How It All Fits Together
**Owner:** team, written together. **Duplicate:** template slide 6.
This is the map slide. It has to go up before anyone builds their own section, because it fixes the order everything else follows.
- Eyebrow: `SYSTEM OVERVIEW`
- Title: `How It All Fits Together`
- Four chevron segments, left to right: `Collect & Clean Data` → `Fine-Tune the Model` → `Backend & API` → `Student & Teacher App`
- One line under each segment: what raw audio gets turned into a clean training set; how the model learns Sinhala speech; how the backend queues and stores transcriptions; how students and teachers actually use it.
- No photo needed. If the template's chevron colors run light-to-dark, keep that direction, since it reads naturally as a pipeline moving forward.

### Slide 6: Building the Dataset
**Owner:** Yohan. **Duplicate:** template slide 2.
- Eyebrow: `DATA COLLECTION`
- Title: `Building the Dataset`
- Column 1 heading `Collect`, body: pulled Sinhala speech from four sources: OpenSLR, YouTube, BizBrains, and Linga.
- Column 2 heading `Clean`, body: removed silent and noisy clips, deduplicated across sources, and stripped invisible characters from the transcripts.
- Column 3 heading `Assemble`, body: combined and manually reviewed everything into one final dataset, ready to split for training.
- Icons: a magnifying glass or funnel for collect, a broom or filter for clean, a stacked-layers icon for assemble, matching the existing line-doodle style rather than flat clip-art.

### Slide 7: Cleaning the Data
**Owner:** Yohan. **Duplicate:** template slide 4.
- Eyebrow: `DATA COLLECTION`
- Title: `Cleaning the Data`
- Column 1: `~67%`, the share of duplicate audio removed from the collected sources once checked against OpenSLR by content, not just filename.
- Column 2: voice activity detection dropped clips that were silence or noise, not speech.
- Column 3: invisible Unicode characters were stripped from transcripts, and flagged rows were checked by hand.
- Key Insight line: cleaning the data took as much work as anything downstream of it, and the model can only be as accurate as what it was trained on.

### Slide 8: Final Dataset
**Owner:** Yohan. **Duplicate:** template slide 11.
- Eyebrow: `DATA COLLECTION`
- Title: `Final Dataset`
- Column 1 heading `Scale`, body: 154,828 rows pooled from two cleaned corpora (OpenSLR plus the combined collection), with an 80/10/10 stratified train/validation/test split, and a separate held-out split for testing generalization.
- Column 2 heading `Quality`, body: every source went through deduplication, noise removal, and manual review of flagged transcripts before being included.

### Slide 9: Baseline Research
**Owner:** Tharupahan. **Duplicate:** template slide 4.
- Eyebrow: `BASELINE RESEARCH`
- Title: `Starting From What Already Works`
- Column 1: built an early proof-of-concept speech demo to check the idea was feasible before committing to it.
- Column 2: evaluated existing published Sinhala ASR models instead of assuming a model would need to be trained from zero.
- Column 3: verified the strongest candidate independently, confirming it transcribes accurately rather than trusting the published number alone.
- Key Insight line: a strong existing model, 10.85% word error rate, gives a proven starting point to fine-tune from instead of starting cold.

### Slide 10: Fine-Tuning Pipeline
**Owner:** Tharupahan. **Duplicate:** template slide 7.
- Eyebrow: `MODEL DEVELOPMENT`
- Title: `Fine-Tuning Pipeline`
- Above the line: `Load Data`, then `Preprocess`. Below the line: `Train`, then `Evaluate`.
- One line under each: pulling in the training set, converting audio into the format the model expects, fine-tuning with LoRA on top of the base model, scoring the result by word error rate so runs are directly comparable.
- Visual: replace the plain timeline with `model-development/diagrams/code-structure.png` from the repo if there's room, since it's the real diagram of this exact flow rather than something redrawn for the slide.

### Slide 11: Connecting to the App
**Owner:** Tharupahan. **Duplicate:** template slide 3.
- Eyebrow: `SYSTEM INTEGRATION`
- Title: `Connecting the Model to the App`
- Main Goal: define one clean handoff so the model and the app can be built independently and plugged together later.
- Supporting Objectives: a single contract, audio in, transcript out; documented exactly where a trained model gets loaded into the backend; a placeholder left in that spot today so the rest of the app isn't blocked waiting on training.
- Photo panel: replace the stock photo with `model-development/diagrams/transcriber-swap-in-point.png` from the repo, the sequence diagram showing today's placeholder and where the real model will load in.

### Slide 12: Quality Check & What's Next for Training
**Owner:** Tharupahan and Yohan together. **Duplicate:** template slide 11.
- Eyebrow: `QUALITY ASSURANCE`
- Title: `Checking Our Own Work`
- Column 1 heading `Self-Audit`: read the project's requirements and design documents and checked the pipeline against them directly, rather than assuming it was compliant. Found a handful of gaps early, like a wrong default model size and a missing accuracy metric, all small fixes now instead of surprises later.
- Column 2 heading `Fine-Tuning: Ready to Launch`: a full GPU fine-tuning setup exists, both a full fine-tune and a LoRA fine-tune of Whisper-small, with experiment logging and ranked evaluation across runs. It hasn't been run for real yet. State this plainly rather than implying a trained model already exists.

### Slide 13: Backend & API
**Owner:** Imindu. **Duplicate:** template slide 9.
- Eyebrow: `APPLICATION`
- Title: `Backend & API`
- Single paragraph: what the backend handles end to end, JWT login, a queue that takes an upload and turns it into a transcription job, a background worker that processes those jobs, and a transcript API for reading, editing, and finalizing the result.
- Photo panel: swap for a screenshot of the interactive API docs (`/docs`) or the database schema/ERD, both of which are real and already exist in the project, rather than a stock photo.

### Slide 14: The App
**Owner:** Imindu. **Duplicate:** template slide 10.
- Eyebrow: `APPLICATION`
- Title: `The App`
- Top photo: a real screenshot of the frontend in use, ideally the transcript review/editing screen since it's the most visual part of the product.
- Three captions underneath: one line each for uploading a lecture, reviewing and correcting a transcript, and taking a spoken quiz.
This is the one slide in the whole deck built around a real photo. Take the screenshot from the actual running app, not a mockup.

### Slide 15: Real Transcription, Wired In
**Owner:** Imindu. **Duplicate:** template slide 8.
- Eyebrow: `APPLICATION`
- Title: `Real Transcription, Wired In`
- What We Created: two real transcriber backends connected to the app, Whisper-small and the SPEAK-ASR Whisper-plus-LoRA baseline, selected by a single config setting.
- What It Does (bullets): accepts an uploaded lecture or quiz answer, queues it for transcription, runs the selected model, and returns text ready for review.
- Photo panel: a screenshot of a completed transcription in the app, or drop the photo and let the text panel take the full width if no clean screenshot is ready in time.

### Slide 16: Challenges Faced
**Owner:** team. **Duplicate:** template slide 11.
- Eyebrow: `CHALLENGES`
- Title: `Challenges Faced`
- Column 1: real duplicate content across data sources meant building a proper cross-source deduplication step rather than trusting each source was independent.
- Column 2: GPU access for a real fine-tuning run has been the bottleneck, which is why training is built and ready but hasn't executed yet.
(Swap in whichever two challenges the team agrees are most worth mentioning; these are the two clearest ones from the repo history.)

### Slide 17: Next Steps
**Owner:** team. **Duplicate:** template slide 12.
- Eyebrow: `NEXT STEPS`
- Title: `Where This Goes From Here`
- Box 1 `Data`: finish the manual QA review still in progress on the combined collection dataset.
- Box 2 `Model`: run the fine-tuning pipeline for real on GPU and compare full fine-tuning against LoRA.
- Box 3 `Backend`: replace the placeholder transcriber with the trained model once it's ready.
- Box 4 `Frontend`: continue building out the teacher review and quiz-grading workflows.

### Slide 18: Thank You
**Owner:** whoever closes. **Duplicate:** template slide 13.
Keep as is.
