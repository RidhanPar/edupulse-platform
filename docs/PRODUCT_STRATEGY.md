# EduPulse: running it as a business

Written from the point of view of the person who owns the product. Everything here is a plan
or a judgement, not a measured result. There are no customers yet, and every model metric in
the app comes from synthetic data.

## The pitch

Upload the spreadsheet you already export from your student records system. Within an
afternoon, advisers see which students are at risk, why, and whose risk has risen since last
week. No integration project, all data held in the EU, and every look at a student's record
is logged.

## Who buys first

**Further education colleges, and single departments or programmes inside universities.**

- They feel retention pressure directly, and often have no analytics team.
- They can run a pilot without a university wide procurement exercise.
- A programme leader can own the pilot personally, which is what makes a pilot happen.

Whole universities come later. They expect single sign on, integrations and a security review
the product cannot pass yet.

## Why someone would choose EduPulse

Established products already exist in this space, for example Jisc's learning analytics
service and Solutionpath in the UK, and EAB and Civitas Learning in the US. Check what each
offers today before positioning against them. EduPulse should not try to out-feature them. It
competes on:

1. **Time to first result.** A CSV and a column mapping screen, not an integration project.
2. **Explainability.** Every prediction shows the indicators behind it, and the model's
   features are named, including the ones the institution did not supply.
3. **Data protection as a feature.** EU hosting, per organisation isolation tested route by
   route, an append only audit log enforced inside the database, no real names required.
4. **Price.** Sized for a college or a single programme, not a whole university.

## What stands between EduPulse and the first paying customer

In order of how hard they block a sale:

1. **It cannot legally hold real student data yet.** It still needs a Data Processing
   Agreement reviewed by a lawyer, a DPIA template the institution can adapt, a breach
   procedure, retention that actually deletes data on schedule, and an independent
   penetration test. Until then the README's "synthetic or anonymised data only" warning
   stays.
2. **Nobody knows whether the model works on real data.** The first pilot must measure it
   (see below). No accuracy figure should appear in sales material before then.
3. **A prediction alone does not retain a student.** The value is in what an adviser does
   next. Without tracking interventions, a pilot cannot show impact.
4. **It is not deployed.** Everything is ready on the `feat/multi-tenant` branch.
   `docs/DEPLOYMENT.md` lists the steps that need an account and a payment method.

## What was built in this round, and why

| Built | Why it matters commercially |
|---|---|
| Production deploy prepared (Render and S3 in Frankfurt, signed models, database enforced audit log, PostgreSQL tests in CI) | A security reviewer's first questions are where the data lives, who can change the logs, and whether stored code can be tampered with |
| Stored prediction runs | Pages read stored results, so a busy results page no longer reruns the model for every viewer. It also gives each student a history |
| "Risk rose since last run" filter | The question an adviser actually asks on a Monday morning |
| Per student page, audited | Where an adviser starts a conversation. Each view is logged, which data protection officers ask for |
| CSV templates and a data dictionary | Cuts the back and forth before a pilot's first upload. It says plainly that names can be pseudonyms |

## Roadmap

**Next: make a pilot possible (needs your decisions)**
- Retention job: purge datasets, prediction runs and audit personal fields on schedule, with
  a dry run mode first. You deferred this earlier. It is now the top blocker for real data.
- Intervention tracking on the student page: assign to an adviser, record contact and
  outcome. Structured outcomes only at first, because free text notes about students are
  sensitive data and widen the DPIA.
- Subject access export: everything held about one student, for one request.
- Compliance pack: DPA, DPIA template, sub-processor list (Render, AWS), breach procedure.

**Then: make a pilot convincing**
- Backtest on last year: train on an older cohort, predict a later one, report precision
  and recall at the threshold advisers will act on.
- Fairness check: the same metrics split by groups the institution chooses to supply.
- Model monitoring: warn when this term's data no longer looks like the training data.

**Later: make it scale**
- Single sign on (Microsoft Entra ID first, since most UK institutions use it).
- Training as a background job, and the rate limiter on Redis so the app can run more than
  one worker.
- Exports from Moodle, Canvas and common student records systems, mapped once per
  institution.

## Pricing approach

Charge per enrolled student per year, banded by the plan tiers already in the database:
pilot (free or heavily discounted, in exchange for a case study and feedback), small (under
1,000 students), medium (1,000 to 3,000) and large (over 3,000). Set actual prices from the
first pilot conversations, not before.

## How a pilot proves value

Agree these before the pilot starts, so nobody marks their own homework afterwards:

1. **Accuracy on their data:** a backtest on last year's cohort, reported at the risk
   threshold advisers will use.
2. **Adoption:** how many flagged students were contacted, and how quickly.
3. **Outcome:** retention of flagged students against the same group the year before.
   This is a comparison, not proof of cause, and should be described that way.

## Risks

- **Labelling students.** A "High risk" badge can become self fulfilling. Advisers should be
  trained to use it to start a supportive conversation, and the wording on the student page
  says so.
- **False positives and false negatives.** Decide with each institution which mistake costs
  more before choosing thresholds.
- **A data incident at a small company ends the company.** That is why the security work came
  before features.

## Decisions needed from you

1. Approve the Render and AWS spend in `docs/DEPLOYMENT.md`, and do the account steps.
2. Approve building the retention job (it deletes data, so it needs your yes).
3. Interventions: structured outcomes only, or free text notes as well?
4. Which institution do you approach first for a pilot?
