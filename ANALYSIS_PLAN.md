# Early-warning models on a later course run: analysis plan

This plan is written before the OULAD data has been downloaded or viewed. Every later change goes in the deviation
log at the end, with its date and reason.

## Questions

1. **Transfer.** How well do the first 2 or 4 weeks of a student's activity predict whether they pass, when the model
   is trained on earlier runs of a course and tested on a later run? How much lower is that than the usual random
   split of the same students?
2. **Fairness.** Is the model equally accurate for every group of students: gender, age band, deprivation level of
   their area and disability?

Early prediction on OULAD has been studied many times, almost always with random splits. The point here is the
honest test (a later run, students not seen in training) and the group gaps under that test.

## Data

OULAD, the Open University Learning Analytics Dataset (Kuzilek, Hlosta and Zdrahal, *Scientific Data*, 2017),
CC BY 4.0. Seven modules, each run in up to four presentations (2013B, 2013J, 2014B, 2014J), about 32,000 students.
Tables used: `courses`, `studentInfo`, `studentRegistration`, `studentVle`, `vle`, `assessments`,
`studentAssessment`. Raw files are checksummed on download and are not committed.

## Who is included

* One row is one student's enrolment in one presentation.
* **Cutoff day** *c* = 14 (primary) or 28 (secondary), counted from the presentation start (day 0), as in OULAD.
* A student is excluded if they unregistered on or before day *c*: by then there is nothing left to predict.
  The number excluded is reported per presentation and group.

## Outcome

`final_result` of Pass or Distinction = 1 (passed). Fail or Withdrawn after day *c* = 0 (at risk).

## Features (only information available on day *c*)

* **Activity:** total clicks; active days; clicks per week; clicks per VLE activity type (the 10 most common types
  in the training data, the rest pooled); days since last activity; clicks before day 0.
* **Assessments:** for assessments due on or before day *c*: whether submitted, score, days late.
* **Enrolment:** days between registration and day 0; studied credits; number of previous attempts; module
  (one-hot).
* **Not used in the primary model:** gender, age band, IMD band, disability, region and highest education.
  They are used only to measure group gaps. A secondary model that also sees them is reported separately.

## Splits

* **Later run (primary):** modules with at least one 2013 and one 2014 presentation, chosen by that rule from
  `courses`, not by name. Train on all their 2013 presentations, test on all their 2014 presentations. A student
  who appears in both years is removed from the test set, so every test student is unseen.
* **Random split (comparison):** the same enrolments, split by student ID (never by row) into train and test with
  the same test fraction as the later-run split. Repeated with 5 seeds; the mean and range are reported.
* No hyperparameter is tuned on either test set. Any tuning uses 5-fold cross-validation inside the training set,
  grouped by student.

## Models (settings fixed now)

1. **Logistic regression:** standardised features, L2 penalty, C = 1.0.
2. **Gradient-boosted trees (primary):** scikit-learn `HistGradientBoostingClassifier`, library defaults,
   random_state = 0.
3. **Sequence model:** a one-layer GRU (hidden size 32) over daily click counts per activity type up to day *c*,
   concatenated with the static features; Adam, learning rate 1e-3, 20 epochs, early stopping on a 10 % grouped
   validation split of the training set; 5 seeds, mean reported.
4. **Majority-class baseline.**

A language-model baseline may be added later. If so, it is labelled post hoc and secondary.

## Metrics

* **Primary:** AUROC of model 2 at day 14 on the later-run test set, with a 95 % CI from 1,000 bootstrap resamples
  of students.
* **Transfer drop:** random-split AUROC minus later-run AUROC, with a paired bootstrap CI.
* **Flagging:** a student is flagged at risk when P(pass) < 0.5. Reported per group: the share of at-risk students
  who are missed (false-negative rate) and the share of passing students wrongly flagged (false-positive rate).
* **Group gaps:** for each attribute, the largest difference in AUROC between its levels, and the largest
  difference in false-negative rate, each with a bootstrap CI.

## Groups

* Gender: as recorded (F, M).
* Age band: 0–35, 35+ (the 35–55 and 55+ bands are merged because 55+ is expected to be very small).
* IMD band: deciles merged into quintiles (0–20 % most deprived ... 80–100 %), with missing values as their own level.
* Disability: Y, N.
* Any level with fewer than 200 test students is merged with its neighbour, and the merge is reported.

## Hypotheses

* **H1.** Later-run AUROC is lower than random-split AUROC. Test: one-sided paired bootstrap, α = 0.05.
* **H2.** The day-28 model has higher later-run AUROC than the day-14 model. Test: paired bootstrap, α = 0.05.
* The group gaps are reported with CIs and no direction is predicted.

## What is reported whatever happens

All four models, both cutoffs, both splits, every group, and the exclusion counts. A result that goes against H1 or
H2 is reported as it is.

## Deviation log

| Date | Change | Reason |
|------|--------|--------|
| | | |
