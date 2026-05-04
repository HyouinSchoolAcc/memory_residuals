# NeurIPS 2026 — Submission Pipeline Tutorial

A walkthrough for first-time NeurIPS authors. Covers the official
2026 dates, the OpenReview mechanics, the paper checklist, the
double-blind rules, the dual-submission rules, supplementary material
constraints, and what to do during reviewing / rebuttal / camera-ready.

This document is not a substitute for the official handbook; it is a
plain-English orientation. Always cross-check against:

- **Main Track Handbook 2026:** <https://neurips.cc/Conferences/2026/MainTrackHandbook>
- **Call for Papers 2026:** <https://neurips.cc/Conferences/2026/CallForPapers>
- **Paper Checklist:** <https://neurips.cc/public/guides/PaperChecklist>
- **Style file:** download `Formatting_Instructions_For_NeurIPS_2026.zip`
  from <https://media.neurips.cc/Conferences/NeurIPS2026/>
- **OpenReview NeurIPS 2026 venue:** the venue page goes live a few
  weeks before the abstract deadline; bookmark
  <https://openreview.net/group?id=NeurIPS.cc/2026>

---

## 0. The 12 things you actually have to do (in order)

If you do nothing else, do these in order. Each is expanded below.

1. **Create an OpenReview profile** for every author. Use your
   institutional email if you have one; institutional emails activate
   immediately, non-institutional ones can take up to **two weeks** to
   moderate. *(One-time per author.)*
2. **Fill out your OpenReview profile completely** — affiliations,
   publications (DBLP import is easiest), conflict-of-interest
   information. **An incomplete profile is grounds for desk rejection.**
3. **Decide track**: Main, Datasets & Benchmarks, or Position Papers.
   Different tracks, different portals, different timelines. **You
   cannot submit the same paper to multiple tracks.**
4. **Register the abstract** by the abstract deadline. This locks
   in: title, authors, area, contribution type, abstract, TL;DR.
5. **Upload the full PDF** by the full-paper deadline (about 48 h
   later). PDF must include the paper checklist *inside* the PDF, or
   the submission is desk-rejected.
6. **Optional: upload supplementary** ZIP (≤ 100 MB), anonymised.
7. **Wait for reviews.**
8. **Author response window**: write rebuttals (≤ 10 000 chars per
   review, no new attachments).
9. **Discussion window**: respond to follow-up reviewer questions.
10. **Receive notification** (accept / reject / borderline).
11. **If accepted**: prepare camera-ready, register at least one author,
    book travel, prepare poster / talk, upload lay summary.
12. **If rejected**: optionally opt-in to make the rejected paper
    public on OpenReview after the 2-week opt-in window.

---

## 1. Key 2026 dates

(All deadlines are **AOE = Anywhere on Earth = UTC−12**, which means
the deadline is when the *very last* place on earth still has that
date. Practically, AOE is "noon UTC the next day".)

| date | what |
|---|---|
| **May 4, 2026 (AOE)** | **Abstract submission deadline** (Main Track) |
| **May 6, 2026 (AOE)** | **Full paper submission deadline** (Main Track) |
| June 6, 2026 (AOE) | Workshop application deadline (organisers, not authors) |
| July 11, 2026 (AOE) | Workshop **acceptance** notifications (workshop-list goes public) |
| August 29, 2026 (AOE) | Suggested workshop **paper** submission deadline |
| September 24, 2026 (AOE) | Author notification (Main Track accept / reject) |
| September 29, 2026 (AOE) | Workshop mandatory accept / reject notification |

**The two deadlines you cannot miss** (for the main track):
**May 4 AOE** and **May 6 AOE**.

If your paper is workshop-bound, you have until **August 29** but
the workshop you target won't be announced until **July 11**, so
budget for ~1.5 months of writing after the workshop list lands.

---

## 2. Tracks (pick one, only one)

1. **Main Track** — general research. Five contribution types: General,
   Theory, Use-Inspired, Concept & Feasibility, Negative Results. Pick
   the one that best fits your paper; reviewers get type-specific
   guidance.
2. **Datasets & Benchmarks Track** — for new datasets and benchmark
   contributions. Different submission portal, separate review pool.
3. **Position Papers Track** — for argument-driven, no-experiments
   papers (or experiments-as-illustration only).

You **cannot submit the same paper to multiple tracks** simultaneously.
Switching tracks after submission is generally not allowed.

**Workshops** are *not* a track — they are separate venues that
piggyback on the conference. Workshop calls go out individually after
July 11. A paper accepted at a workshop is *non-archival* (no
proceedings), so a workshop paper can be re-submitted to the next
year's main track without dual-submission concerns. **Main-track papers
*are* archival** (they appear in the proceedings).

### Contribution types for the main track (2026)

Pick on the OpenReview submission form. Reviewers get type-specific
guidelines:

- **General** — most papers fall here. Default if unsure.
- **Theory** — main contribution is via analyses and proofs.
- **Use-Inspired** — main contribution is novel methods / tasks /
  metrics tied to a real-world use case.
- **Concept & Feasibility** — high-risk, high-reward idea with
  preliminary results. Bar is high.
- **Negative Results** — main contribution is in understanding a
  negative result. Bar is high. Suitable for our Paper C.

---

## 3. OpenReview profile setup (per author, one-time)

1. Go to <https://openreview.net> → **Sign Up**.
2. Use **your real legal first and last name** as they appear on your
   institutional records. OpenReview generates a profile slug like
   `~Firstname_Lastname1` from this. The trailing `1` is a counter; if
   someone with the same name already has `~Firstname_Lastname1`, you
   get `~Firstname_Lastname2`, etc. **Write down your slug** — you'll
   paste it into the submission form.
3. Use your **institutional email** if you have one. New profiles
   created with non-institutional emails are moderated by humans and
   can take **up to two weeks**. If you only have a personal email, do
   this step at least three weeks before the deadline. **As of
   2026-05-04, with a deadline of May 4 AOE, a non-institutional
   profile may not finish moderation in time.**
4. Add affiliations under **Education & Career History**. Include all
   substantive ones from the last 3 years (full-time employment,
   sabbaticals, ≥20 % consulting, etc.). These drive **domain
   conflict-of-interest** detection.
5. Add publications (DBLP import is easiest, but optional — you can
   manually add or leave empty if no DBLP page).
6. **Add personal conflicts** under "Advisors & Other Relations":
   PhD advisors / advisees, family, all co-authors of original
   research from the last 3 years. **A profile that does not
   declare conflicts can result in desk rejection.**
7. Verify your preferred contact email. All OpenReview email comes
   from `noreply@openreview.net` — whitelist this if your spam filter
   is aggressive.

> **Practical:** for our three concurrent submissions, the two authors
> are `~Yueze_Liu1` and `~Ajay_Kumdam1` (placeholders — verify the
> trailing counter once each profile is registered).

---

## 4. Anatomy of an OpenReview submission form

The form is identical across the three Main Track papers; you fill it
out three times. Field-by-field:

| field | what it expects |
|---|---|
| **Title** | The paper's working title. Camera-ready can edit if no substantial change. |
| **Authors** | Add by OpenReview ID (`~Firstname_Lastname1`). All authors must be added before the **paper deadline**; only the order can change after that. |
| **TL;DR** | 1–3 sentences. Visible to reviewers as the first thing they read. |
| **Abstract** | Plain-text abstract. Often capped at **1750 characters** (~ 250 words) — pre-trim before pasting. |
| **Primary Area** | Pick from the dropdown (categories like "Foundation or Frontier Models", "Deep Learning", "Empirical analysis"). |
| **Secondary Area** | Optional second category. |
| **Contribution Type** | One of: General, Theory, Use-Inspired, Concept & Feasibility, Negative Results. |
| **Keywords** | Comma-separated list. Used for reviewer matching. |
| **PDF Upload** | Single PDF, ≤ 50 MB. Must include the paper checklist inside the PDF. |
| **Supplementary** | Optional ZIP, ≤ 100 MB. Anonymised. |
| **Checklist Confirmation** | Tickbox: "I confirm that the paper checklist is included in the paper PDF." |
| **Reviewer Nomination** | Optional self-nominated qualified reviewer (an OpenReview ID). |
| **Responsible Reviewing** | Tickbox acknowledgement. |
| **Academic Integrity** | Tickbox acknowledgement. |
| **Declaration** | Tickbox confirming submission complies with policies. |
| **License** | Pick the suggested CC license; the default is fine for nearly all papers. |
| **Financial Support** | Skip unless you're a student needing travel-grant consideration. |
| **LLM Usage** | Tickboxes for what role LLMs played: writing assist, code assist, ideation, primary methodology. **Be honest** — see §10 below. |
| **LLM Experiment** | Optional opt-in to a 2026 experimental review track that uses LLM-assisted reviewing on opted-in papers. Default opt-out. |

The abstract registration submission is the one where you **first
fill out almost all of these fields**. The full-PDF submission lets
you edit the same record and upload the actual PDF.

> **Practical:** keep a single text file with all 18 standing answers
> for our papers (we already have this in `NEURIPS_SUBMISSIONS.md`).
> Three submissions × ~5 minutes each ≈ 15 minutes of form filling on
> the abstract day, if everything is pre-drafted.

---

## 5. PDF requirements

Single PDF, ≤ 50 MB, containing in order:

1. **Main paper content** — limited to **9 content pages**, including
   all figures and tables.
2. **References** — unlimited.
3. **Appendices** — unlimited.
4. **The NeurIPS 2026 paper checklist** — pasted from the LaTeX
   template. **Mandatory.** If absent → **desk rejected**.

**Style file:** download
`Formatting_Instructions_For_NeurIPS_2026.zip` from the NeurIPS 2026
website. It is a `\usepackage{neurips_2026}` package similar to prior
years. **The Word template is discontinued** — LaTeX only.

**Anonymity:** the PDF is double-blind. Strip:

- author names (use the placeholder `Anonymous Authors`)
- affiliations
- acknowledgements (add at camera-ready)
- self-citations that reveal authorship ("In our previous work [1]" →
  "In Smith et al. [1]"), or ("Anonymous et al. [1]") if the cited
  paper is concurrently in submission

If you have a *non-anonymous preprint* on arXiv, that is allowed (and
not a violation), but **do not aggressively advertise** the preprint
during the review period — that can be flagged as a double-blind
violation.

**Compute reporting:** authors will be asked to report compute
resources used. This is separate from the paper checklist and does not
count for review; it is an information-gathering question.

---

## 6. The NeurIPS 2026 paper checklist (mandatory)

Pulled from the LaTeX template at the end of your PDF. Roughly 15
yes/no/n-a questions covering claims, limitations, theory,
reproducibility, code/data, experimental details, statistical
significance, compute, ethics, broader impacts, safeguards (for
released models / data), licenses, contributions to existing assets,
new assets, crowdsourcing, IRB approval, declaration of LLM usage in
research method (NEW in 2026).

For each question:

- Answer `[Yes]`, `[No]`, or `[N/A]`.
- Add a 1-2 sentence justification, especially when answering "No" or
  "N/A". Reviewers read the checklist; "no" with a good justification
  is fine and is **not** automatic grounds for rejection.
- Reference the specific section / appendix that backs the answer.

> Practical: copy the entire `\section*{NeurIPS Paper Checklist}` block
> from the template, fill out each `\answer{}` with `[Yes]/[No]/[N/A]
> + your 1-2 sentence justification`, leave the surrounding question
> text untouched (it must match exactly or reviewers will flag it).

> **Common mistake:** people leave the checklist section in but answer
> every question with the placeholder `[TODO]`. That counts as "did
> not answer" and reviewers will note it.

---

## 7. Supplementary material

Optional. Single ZIP, ≤ 100 MB, anonymised.

- **Code:** include training + evaluation code, dependency spec,
  README with reproduction instructions. Anonymise everything: no
  GitHub URLs that resolve to real accounts, no `author = "..."` in
  pyproject.toml, no `__pycache__` or `.pyc` (these can carry
  filesystem paths from your machine).
- **Data:** if you're releasing a dataset, follow the data submission
  guidelines. If your dataset is publicly available (e.g., LongMemEval-S),
  cite it; do not redistribute the corpus.
- **Eval JSONs / numerical artifacts:** including these is encouraged.
  Reviewers will read them only if curious, but they support
  reproducibility claims.
- **Model checkpoints:** generally too large to include. Mention in
  the appendix that they will be released under an anonymised
  archive at camera-ready time.

> **Practical:** our three papers have anonymised supplementary zips
> already built — `memres_supplementary.zip`,
> `failure_modes_supplementary.zip`, `pair_recipe_supplementary.zip`.
> Each has a README with reproduction instructions and per-cell
> launcher scripts.

---

## 8. The double-blind rules (the part that gets people desk-rejected)

The PDF, the supplementary, and any links must all preserve
double-blindness.

| do | don't |
|---|---|
| Strip author names, affiliations, acknowledgements from PDF | Leave a `\author{Yueze Liu}` block in the .tex |
| Use `Anonymous Authors` / `anonymous@example.com` placeholder | Self-cite as "in our prior work [1]"; instead "Smith et al. [1]" |
| Anonymise GitHub URLs to a per-paper anonymised mirror, or a `<anonymised>` placeholder | Link to a public GitHub repo bearing your name |
| Strip metadata from PDFs (`pdftotext` to verify; LaTeX `\nonstopmode \pdfinfo {...}` etc.) | Submit a PDF whose `pdfinfo` reveals author info |
| If you cite a concurrent submission of yours, write "Anonymous et al. [1] concurrently show..." and include the cited submission as a separate file in the supplementary | Cite a concurrent submission by name |

Authors of *non-anonymous preprints* on arXiv: that is permitted, but
do not link or reference the preprint in the submission, and **do not
heavily advertise** the preprint during the review period.

---

## 9. Dual-submission rules

You **may not**:

- Submit the same paper to NeurIPS *and* another archival venue
  simultaneously.
- Submit two papers to NeurIPS that overlap so heavily that publishing
  one renders the other "too incremental" — this is called
  "thin-slicing" and **both** can be rejected.
- Submit the same paper to both Main and DB tracks.

You **may**:

- Submit a paper that has been presented at a *non-archival* workshop.
- Submit a paper while a non-anonymous preprint is on arXiv (preprint
  ≠ archival venue).
- Submit two papers with overlapping methods if the contribution
  claim is clearly different. **Be explicit** in the introduction:
  cite the concurrent submission anonymously and note which
  sub-claim each paper carries. Reviewers will appreciate the
  honesty; without it, both papers risk dual-submission flags.

> **Practical:** our three Memory Residuals papers (P1 / Paper A /
> Paper B / Paper C) all share the same architecture. The contribution
> split must be explicit:
>
> - **Paper A** (v28 + RAG): empirical claim that recurrent memory
>   beats RAG on the LongMemEval-S callback-CE metric at 0.6B and 1.7B.
> - **Paper B** (attention parity vs simple gate): architectural
>   claim that the parity-preserving init is load-bearing and ~2×
>   sample-efficient over a ReZero baseline. Different training cell
>   (PG-19 + TV) and different metric (TBPTT next-token NLL).
> - **Paper C** (negative results / ideologies): methodology
>   contribution — the six-item audit battery and the three-ideology
>   framing. Numerical content is post-hoc audits of v11–v28.
>
> Each paper's §1 should include a 2–3-sentence concurrent-submission
> note pointing to the others (anonymised: "Anonymous et al. [X]
> concurrently report ...").
>
> NeurIPS auto-flags any author listed on **4 or more** papers as a
> reviewer commitment. Yueze + Ajay are at 3 each across these
> submissions. Do not add a fourth without budgeting reviewer time.

---

## 10. LLM usage declaration

NeurIPS 2026 explicitly asks about LLM use during paper writing,
coding, and methodology development. **Be honest** — declaration is
required, not optional, and is publicly visible.

The form has tickboxes:

- **Writing assistance** — using LLMs to phrase or polish prose. Tick
  if used.
- **Code assistance** — using LLMs (Copilot, Cursor, Claude, etc.)
  for code generation. Tick if used.
- **Research assistance / ideation** — using LLMs for brainstorming
  or research design. Tick if used.
- **LLM as the methodology** — if your paper *uses* an LLM as a
  central component of the method (e.g., LLM-as-judge), this is a
  separate declaration in the experimental setup section.

There is also a free-text **Other LLM Usage** field (~1–2 sentences)
where you describe the role honestly.

> **Practical:** our standing answer for all three papers (per
> `NEURIPS_SUBMISSIONS.md`):
>
> > "Cursor agent (Claude Opus 4.7) used throughout the project for
> > code generation, experiment scripting, run-log analysis, and
> > manuscript drafting; all numbers in the paper come from
> > agent-launched but human-verified runs against the locked corpus
> > and eval script."

The 2026 chairs note that **prompt-injection attempts** in submissions
(text optimised to manipulate LLM-assisted review) are strictly
prohibited and will be rejected on sight.

---

## 11. Submitting — actual click-through walkthrough

Assuming you have a profile and a PDF ready:

1. Sign in to <https://openreview.net>.
2. Navigate to the NeurIPS 2026 venue page
   (<https://openreview.net/group?id=NeurIPS.cc/2026/Conference>).
   The "Submit" button appears once the abstract submission window
   opens (typically a few weeks before the abstract deadline).
3. Click **New Submission**. The form appears.
4. Fill in the fields per §4 above. Save as you go (OpenReview keeps
   a draft).
5. **Author addition:** type the OpenReview slug or email of each
   co-author into the Authors field; the form auto-completes from
   their profile. **Add yourself first**, in the order you want
   the author list rendered.
6. Upload the PDF (50 MB cap). The system runs an automatic format
   check; minor formatting violations are warnings, major ones
   (missing checklist, exceeded page count) cause the submission to
   be rejected.
7. **Submit** the form by the abstract deadline. Your title /
   authors / abstract / area are now locked, but you can still
   replace the PDF up to the **full-paper deadline** 48 h later.
8. Optional: upload supplementary ZIP (separate field, may have a
   later deadline depending on year).
9. Verify the submission appears on your "Authored Submissions"
   page on OpenReview.

> **Practical:** the abstract registration is what you cannot miss.
> If your full PDF is not ready, upload a *stub* PDF with title,
> abstract, and a section skeleton — that satisfies the abstract
> deadline. Then replace the stub with the real PDF before the
> full-paper deadline.

---

## 12. After submission: the review timeline

| phase | when | what you do |
|---|---|---|
| **Submitted** | May 6 → review window opens | nothing — wait. Don't post the paper to social media (low-key arXiv preprint is OK; aggressive promotion can be flagged as a double-blind violation) |
| **Reviews come in** | ~early July (typical) | reviews appear under your submission in OpenReview; you cannot respond yet |
| **Author response window** | ~3 days, usually mid-late July | you can post a response to each review (≤ 10 000 chars per review). No new files; rebut with text only |
| **Author–reviewer discussion** | another ~5 days | reviewers ask follow-ups; you can respond. Reviewers also discuss with each other |
| **Reviewer-AC discussion** | final phase | authors no longer have visibility |
| **Notification** | September 24, 2026 (AOE) | accept / reject / borderline-with-meta-review |

### How to write a rebuttal that helps

1. **Lead with the one or two highest-impact responses.** Reviewers
   may not read past the first 1000 characters carefully.
2. **Address the AC's meta-review first.** The AC's initial meta-review
   summarises what they think is the *deciding* concern; addressing
   that successfully is more important than rebutting every minor
   reviewer comment.
3. **For new experiments / numbers**, be explicit: "We ran experiment
   X during the rebuttal; the numbers are: ...". Reviewers can update
   their scores on this basis.
4. **Quote the reviewer**, then respond. This makes it clear which
   point you're addressing.
5. **Don't be defensive.** Even when a reviewer is wrong, frame the
   response as "We see how the description led to this confusion; we
   will clarify in the camera-ready by ...". Reviewers respond well
   to acknowledgements.
6. **Per-review limit is 10 000 characters.** Use Markdown formatting
   (OpenReview supports a subset). No file uploads, no figures.

---

## 13. After acceptance: camera-ready and presentation

If the paper is accepted:

1. **Camera-ready PDF**: due ~3-4 weeks after notification. You get
   one extra content page (10 pp main + appendix + checklist + funding
   statement). Edit the PDF in response to reviewer comments. Upload
   via OpenReview's "Camera Ready Version" button.
2. **De-anonymise** the camera-ready: add author names, affiliations,
   acknowledgements, funding statement, competing-interests
   disclosure.
3. **Lay summary** (NEW for 2026): you'll be asked to upload a
   paragraph-length summary aimed at the general public, no jargon.
4. **Author registration**: at least one author must register for the
   in-person main conference. A virtual-only pass is **not**
   sufficient. If the registering author is a student, they need only
   a student pass.
5. **Poster** (most accepted papers go to poster): poster
   specifications depend on the venue and are posted closer to the
   conference. Standard sizes are ~36" × 48".
6. **Talk** (a small fraction of accepted papers get spotlight or oral
   slots): follow accessibility guidelines for slides.
7. **Code release** (encouraged): de-anonymise GitHub mirrors, finalise
   licenses, link from camera-ready PDF.

If the paper is rejected:

1. You get the reviews and meta-review for archival.
2. **Within 2 weeks of notification**, you can opt-in to making the
   rejected paper public on OpenReview (de-anonymised). It will be
   marked "Rejected" but is searchable. Many people choose to opt-in
   for visibility / preprint citability.
3. Most rejected papers re-submit to ICLR 2027 or a workshop / next
   year's NeurIPS. The reviews you got are valuable — incorporate
   them honestly.

---

## 14. Workshop submissions (separate process)

Workshops are confirmed on **July 11**. Each workshop has its own
call, deadline, and OpenReview venue (or sometimes other systems).

Typical workshop pipeline:

1. **July 11**: workshop list published.
2. **July 11 → late August**: workshops open submissions; deadlines
   typically around **August 29**.
3. **Submit** to the workshop's OpenReview venue. Workshop submissions
   are **non-archival** (no proceedings) and have shorter format —
   typically 4-pp main with extended abstract format, sometimes 6-pp.
4. **September 29**: mandatory accept/reject notification.
5. **December 6-12, 2026 (conference dates)**: workshop happens at
   the conference. Most workshops have invited talks + poster session
   + a small number of contributed talks.

**Why workshop:** lower bar than main track, faster reviewing, and
the work stays non-archival, so you can submit a polished version to
the next year's main track.

**Workshop fit for our three papers** (assuming the workshop list
follows past trends):

- **Paper A (v28 + RAG)**: a "Long-Context / Memory in Foundation
  Models" or "Foundation Model Efficiency" workshop is ideal.
- **Paper B (parity vs gate)**: same long-context/memory workshop, or
  a "Foundation Model Architecture" workshop.
- **Paper C (negative results)**: "I Can't Believe It's Not Better!"
  (the canonical negative-results workshop, runs every year) is the
  natural home.

---

## 15. The most common ways to get desk-rejected

A non-exhaustive list, all of which are well-documented in the
handbook:

1. **No paper checklist in the PDF.** Mandatory. Pasted into the
   end of the PDF, not just answered on the form.
2. **Page-limit violation.** > 9 main content pages (excluding
   references / appendix / checklist). Even minor violations are
   flagged; major ones desk-reject.
3. **Style file violation.** Smaller margins, smaller fonts. Use the
   provided LaTeX template unmodified.
4. **Author info in the PDF.** "Yueze Liu, S;G studio" left in.
   Authors who self-cite as "in our prior work" without anonymisation.
5. **Incomplete OpenReview profile.** No affiliations, no conflicts.
6. **Dual submission.** Same paper to two venues, or two
   thinly-sliced versions to NeurIPS.
7. **Plagiarism / undisclosed reuse of others' code or text.**
8. **Prompt injection** in the PDF aimed at LLM reviewers.
9. **No declaration of significant LLM use** when LLMs are central
   to the methodology.

---

## 16. Quick reference — what to do *right now* (May 4, 2026, T-minus 14h to abstract deadline)

Today (May 4, EST evening):

1. Both authors verify their `~Yueze_Liu1` and `~Ajay_Kumdam1`
   OpenReview profiles are activated and their conflicts are filled
   in. (If not activated by tomorrow morning, that author cannot be
   added to the form — desk-reject risk.)
2. Pull the LaTeX style file from
   <https://media.neurips.cc/Conferences/NeurIPS2026/>.
3. Skim `PAPER_A_v28_RAG.md`, `PAPER_B_attention_parity.md`,
   `PAPER_C_negative_results.md` to confirm titles, abstracts, and
   contribution types are what you want to lock in tomorrow.

Tomorrow morning (May 5, before the AOE deadline):

1. Open the OpenReview NeurIPS 2026 venue submission portal.
2. Submit the abstract form for **Paper A**, with a stub PDF (title +
   abstract + section skeleton). ~5 min.
3. Same for **Paper B** and **Paper C**. ~5 min each.
4. Verify all three submissions appear under "Authored Submissions"
   on each author's OpenReview profile.

Then May 5 → May 6 (T-2 days):

1. Finish the three PDFs. The bulk of the work is on Paper A (most
   re-framing relative to existing material), Paper C (most new
   writing for the three-ideology framing). Paper B is mostly a 12 →
   9 page trim.
2. Paste the NeurIPS 2026 paper checklist into each PDF. Fill out
   every question (Yes / No / N/A + 1-2 sentence justification).
3. Re-run anonymity checks on each PDF (`pdftotext` for author
   leaks, `unzip -l` on each supplementary zip for path leaks).

May 6 (full-paper deadline AOE):

1. Upload the final PDF for each of the three submissions.
2. Upload each supplementary zip.
3. Verify the OpenReview "Submitted" status for each.

May 6 → September 24:

- Don't promote the papers heavily.
- Don't refresh OpenReview compulsively (you'll see reviews when
  they're published).
- Use the time to do follow-up experiments that you can fold into
  the rebuttal if reviewers ask for them.

---

## 17. Where to ask for help

- **OpenReview technical issues** (login, profile, can't submit):
  `info@openreview.net`.
- **NeurIPS-specific submission questions**: `program-chairs@neurips.cc`
  (use sparingly; check the FAQ first).
- **Conflict-of-interest concerns**: handled via your OpenReview
  profile's hidden-conflict mechanism (visibility = "NeurIPS 2026
  Program Chairs" only).
- **Ethics concerns / harassment / scientific integrity violations**:
  `scientific-integrity-chairs@neurips.cc` or `hotline@neurips.cc`
  (Code of Ethics violations).
- **Within Cursor** (this codebase's tooling): re-read
  `NEURIPS_SUBMISSIONS.md` for our pre-filled standing answers, and
  `PAPER_A_v28_RAG.md` / `PAPER_B_attention_parity.md` /
  `PAPER_C_negative_results.md` for per-paper plans.

---

## 18. The minimum-viable path if you only have one weekend

If you read this on the morning of the abstract deadline and you've
done none of the prep:

1. **STOP**. Register OpenReview profiles for everybody right now.
   Use institutional emails.
2. **Pick one paper** and submit only that. A clean single submission
   is better than three rushed ones.
3. **Use a stub PDF** for the abstract submission (title, authors,
   abstract, section headings). The full PDF deadline is 48 h
   later — you have time.
4. **Skip the checklist for the abstract submission**; it's required
   for the *full PDF*, not the abstract registration.
5. **Then sleep**. The next 48 hours are about the PDF, the
   checklist, and anonymity-checking. None of that is doable on a
   sleep deficit.

The rule of thumb: an abstract registration with stub PDF takes 5-10
minutes once your profile is set up; **only your profile-setup time
is irreversible**. Everything else can be fixed later.
