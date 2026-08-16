# Safety and regulation

Research date: **2026-08-08**. Question: for an indoor commercial deployment of
a mobile manipulator in the US and EU, what actually applies, what does a
supervised pilot require, and what does it cost?

> **TL;DR.** One architectural decision dominates everything else: **the machine
> learning policy must never perform a safety function.** Get that right and you
> self-certify for **€1.5–5k**; get it wrong and the EU Machinery Regulation
> forces a **Notified Body — €5–15k+ and 3–6 months** — and, since Regulation
> (EU) 2026/1744, the AI requirements arrive *through the Machinery Regulation
> itself* rather than as a parallel AI Act assessment. **One decision, one
> regulation, one assessment.** It is also a description of the Tier 0
> architecture this repository already has.

> **Re-verified 2026-08-15.** The EU mechanism changed between the research
> date and the re-check: the Digital Omnibus (Regulation (EU) 2026/1744, in
> force **2026-07-27**) moved the Machinery Regulation from Section A to
> Section B of the AI Act's Annex I — see the dated update in §1 and the
> settled timeline in §8. The engineering advice survives unchanged; the
> description of *how the two laws interact* did not. Regulatory claims decay
> fastest of anything in this doc set.

⚠️ **This document is engineering research, not legal advice.** Several cost
figures come from compliance-consultancy marketing pages rather than Notified
Bodies, and are labelled where that is so. Get real quotes and a lawyer.

---

## 1. The decision

**EU Machinery Regulation (EU) 2023/1230** was adopted 2023-06-14 and
**applies from 2027-01-20 with no transition period** — it replaces the
Machinery Directive 2006/42/EC outright on that date. That is roughly 17 months
from this research date. (Re-verified at EUR-Lex 2026-08-15: the date stands; a
corrigendum fixed a clerical error without moving it, and the regulation has
been amended by (EU) 2024/2748 — internal-market emergency procedures — and by
(EU) 2026/1744, on which see below.)

Recital 12 explicitly brings *"artificial intelligence, the Internet of things
and robotics"* into scope. Article 3(3) makes safety components expressly
include **digital devices and software**. Article 3(16) makes modifications
*"by physical or digital means"* that create new hazards a **substantial
modification** requiring re-assessment — which is a live concern for a system
that ships new model weights.

And then **Recital 54 / Annex I Part A**:

> Mandatory **Notified Body** conformity assessment for at least six high-risk
> categories, including **"safety components with fully or partially
> self-evolving behaviour using machine learning"** — justified by *"data
> dependency, opacity, autonomy and connectivity."*

**EU AI Act Article 6(1)**, as the regime stood at this doc's research date,
had **two cumulative conditions** for an AI system in machinery to be
high-risk: (1) the AI is used as a **safety component** of a product covered by
Annex I Union harmonisation legislation, **AND** (2) that product is
**required to undergo third-party conformity assessment**. Because condition
(2) referred back to the Machinery Regulation, staying out of Annex I Part A
also dropped you out of AI Act high-risk classification — two regimes, one
escape hatch.

The European Commission's worked example of a safety component was
**"vision systems detecting human presence in robot cells to trigger safe
stops."** That is exactly the thing not to build with a vision-language-action
model. (⚠️ Not re-verified 2026-08-15, and any Art. 6(1) guidance should now be
re-read against the update below.)

### Updated 2026-08-15: the Omnibus restructured the mechanism

**Regulation (EU) 2026/1744** (the Digital Omnibus, CELEX 32026R1744, in force
**2026-07-27** — enacted law, not a proposal) **moved the Machinery Regulation
from Section A to Section B of the AI Act's Annex I**, its recital calling this
"a sectoral approach." The mechanics that follow, from the enacted text:

- For Section B products, amended AI Act Art. 2(2): **"only Article 6(1),
  Article 60a and Articles 102 to 112 shall apply"** — the AI Act's full
  Chapter III high-risk regime no longer applies directly to AI in machinery.
- Instead, a new third paragraph of **Machinery Regulation Art. 8** requires
  the Commission to adopt **delegated acts amending MR Annex III** to add
  health-and-safety requirements for high-risk AI systems (reflecting AI Act
  Ch. III §2 and Arts. 17, 19, 72, 73). **Those delegated acts apply by
  2028-08-02.**
- New **MR Art. 20(10)**: until MR-side harmonised standards or common
  specifications exist, compliance with standards harmonised under AI Act
  Arts. 40/41 gives **presumption of conformity** with the MR's AI
  requirements.

What this changes and what it does not: the old "two cumulative conditions"
escape-hatch description above is **superseded** — the consequence of putting
ML in a safety function is now delivered *through the Machinery Regulation
itself* (delegated-act requirements assessed by the machinery Notified Body),
not through a parallel AI Act Chapter III conformity assessment. The
engineering decision is unchanged and its payoff is **cleaner than before**:
keep the policy out of safety functions, stay out of MR Annex I Part A, and
there is one regulation and one (self-)assessment to face rather than two
regimes to reason about.

### What that means concretely

> - The policy **commands motion**. It **never** commands or inhibits a safety
>   function.
> - Safety-rated speed and force limits are enforced **below** the policy, in
>   hardware or in a certified safety controller, and the policy **physically
>   cannot exceed them**.
> - Human detection for protective stops uses a **certified safety scanner
>   (PLd)** — not your vision model. You may *also* run a vision model, but only
>   as additive comfort, never as the safety case.
> - **Document this separation explicitly in the technical file.** The argument
>   is the deliverable.

This is a description of the four-tier architecture in
[19-the-system.md](19-the-system.md): Tier 0 is deterministic, small, `no_std`,
allocation-free, `#![forbid(unsafe_code)]`, and cannot be overridden from above.
That work was done for engineering reasons. It turns out to have a regulatory
value nobody was designing for.

The corollary is a constraint on future work: **keep Tier 0 small and
independently verifiable.** Every capability added to it has to be justified to
someone who will read it.

---

## 2. The standards, and the gap

| Standard | Status | What it covers |
|---|---|---|
| **ISO 10218-1:2025 / -2:2025** | published **Feb 2025**, first major revision since 2011 | Part 1 robot hardware, Part 2 applications and cells. **Absorbs ISO/TS 15066** — power/force limiting and speed/separation monitoring are now normative. Adds robot classifications with functional safety requirements, and **safety-related cybersecurity**. ⚠️ **Assumes statically stable robots.** |
| **ANSI/A3 R15.06-2025** | Parts 1 & 2 approved 2025-08-21, published Sept 2025, 403 pages | US adoption of ISO 10218:2025 |
| **ISO 3691-4:2023** | current; **ISO/DIS 3691-4 in draft** | Driverless industrial trucks — AGVs, **AMRs**, automated guided carts. ⚠️ **Does not clearly address AMRs with manipulators.** |
| **ANSI/A3 R15.08** | Part 1 (2020), Part 2 (2023), **Part 3 (2026)** | Industrial mobile robots. **Part 3 targets users and operators** — which is what your customer will be asked about |
| **ISO 25785-1** | **still under development** (unrefuted but ⚠️ unverifiable 2026-08-15 — iso.org refused automated access; re-check by hand) | Dynamically stable industrial mobile robots |
| **ISO 13482** | under revision; **ISO/FDIS** described as *"nearing finalization after twelve years"* (2026-05-20) | Personal care / service robots. Wrong standard for a warehouse or lab; **becomes relevant for a clinic** — and its unsettled state is a reason to prefer warehouse or lab for a first pilot |
| **ANSI/CAN/UL 3300:2024** | first published 2023 | Service, communication, information, education and entertainment robots. **Now on OSHA's NRTL list of appropriate test standards**, which is what makes it commercially necessary in the US. ⚠️ Its scope says "does not require instructed or skilled person intervention during operation" — a supervised pilot arguably is not that, so the fit is imperfect |
| **ISO 13849-1:2023** | current | Performance levels for safety-related control systems |

The most important conceptual change in ISO 10218:2025:

> **Collaboration is redefined as "a property of the application, not the
> robot."**

There is no such thing as buying a collaborative robot and being done. *Your*
deployment — this arm, this end-effector, this payload, this workspace, these
people — is what gets assessed.

### What officialdom has actually published for ML safety components (2026-08-15)

The first official Notified-Body-side guidance now exists. **RfU
CNB/M/00.514 Rev 02** (Recommendation for Use, dated 2025-11-28, endorsed by
the Machinery Expert Group 2025-11-07), from the European Co-ordination of
Notified Bodies for Machinery's ad-hoc working group on **exactly the ML
categories of MR Annex I Part A items 5 and 6**, resolves the who-assesses-what
question: a machinery Notified Body **"shall be entitled to control the
conformity of the high-risk AI systems"** (wording from AI Act Art. 43(3)),
must itself comply with AI Act Art. 31(4), (5), (10) and (11), and **need not
be a full AI Act Notified Body**. ⚠️ The RfU predates Regulation 2026/1744 and
reflects the old Section-A mechanics — its procedure may be superseded by the
sectoral approach in §1; watch for a Rev 03. Same pack: **RfU CNB/M/00.518** —
EU type-examination certificates issued under 2023/1230 before 2027-01-20 are
valid five years from issue.

The rest of the official scaffolding, as of 2026-08-15, is scaffolding only:

- A **standardisation request** supporting MR 2023/1230 exists (Commission
  Implementing Decision **C(2025)129**), but **no harmonised-standards listing
  under 2023/1230 has appeared in the Official Journal** — the Commission's
  standardisation page still points only at Directive 2006/42/EC standards.
- **No official application guide for the MR exists** — only the Machinery
  *Directive* guide, edition 2.3 (April 2024).
- NANDO has moved into the Single Market Compliance Space; MR Notified-Body
  designations are implied but were not enumerable by automated access, and
  **whether any body is designated for the Part A ML items specifically is
  unverified**.

### The mobile-manipulator gap is real, and you should name it first

ISO 10218 covers the arm and assumes static stability. ISO 3691-4 covers the
driverless truck. **Neither cleanly covers "mobile robot plus arm,"** and the
standard that would (ISO 25785-1) is still in development.

**Practical consequence: you will perform a bespoke ISO 12100 risk assessment
and justify your choice of standards.** Say this to the customer's EHS lead
before they discover it themselves — arriving with the gap already identified and
a documented assessment is a completely different conversation from being caught
by it.

---

## 3. Stops, and what "safety-rated" means

**Performance Levels** (ISO 13849-1) are defined by probability of dangerous
failure per hour:

| PL | Dangerous failures per hour |
|---|---|
| a | ≥10⁻⁵ to <10⁻⁴ |
| b | ≥3×10⁻⁶ to <10⁻⁵ |
| c | ≥10⁻⁶ to <3×10⁻⁶ |
| **d** | **≥10⁻⁷ to <10⁻⁶** |
| **e** | **≥10⁻⁸ to <10⁻⁷** |

A Performance Level is achieved by a *combination* of architectural Category,
mean time to dangerous failure, diagnostic coverage and common-cause-failure
analysis — **not by Category alone**. Categories run B (basic principles), 1
(well-tried components), 2 (periodic checking), 3 (tolerates a single fault),
4 (single-fault tolerant with comprehensive fault detection).

**Typical robot e-stop target: PLd or PLe.** The canonical architecture is
**PLd Category 3** — dual-channel demand paths with cross-monitoring, so no
single failure of either channel loses the safety function.

**Stop categories** (IEC 60204-1):

- **Category 0** — immediate removal of actuator power; uncontrolled coast or brake
- **Category 1** — controlled stop *with power retained to achieve the stop*, then power removed
- **Category 2** — controlled stop with power retained; the robot stays energised and holds position
- **Safety-rated monitored stop** — a Category 0 or 1 stop where **the stopped
  state is actively monitored**; drift beyond a threshold triggers a protective
  stop. This is what lets a human enter the space while the robot is "on."
- **Protective stop** — typically triggered by laser scanners, time-of-flight
  cameras or light curtains. Categories 0, 1 or 2 allowed, and **reset can be
  automatic**.

The practical distinction: **an e-stop is Category 0/1 and requires manual reset;
a protective stop can be Category 2 and can auto-reset.** For a mobile
manipulator you want both — a hard-wired Category 0/1 e-stop at PLd Category 3,
plus scanner-triggered protective stops so the robot can resume without someone
walking over.

---

## 4. The United States

**OSHA states plainly, on osha.gov/robotics/standards:** *"There are currently
no specific OSHA standards for the robotics industry."*

What applies instead, from 29 CFR 1910: Subpart D (walking-working surfaces),
Subpart G (noise), Subpart I (PPE), **Subpart J — lockout/tagout, 1910.147**,
**Subpart O — machinery and machine guarding**, Subpart S (electrical).

Enforcement runs through the **General Duty Clause**, Section 5(a)(1), applied
via voluntary consensus standards. In practice:

> **The consensus standards are legally voluntary but evidentially decisive.**
> If someone is injured and you deviated from R15.06 or R15.08 without a
> documented justification, the citation writes itself.

⚠️ Note OSHA's own page still cites **ANSI/RIA R15.06-2012**, not the 2025
edition.

Expect **lockout/tagout (1910.147)** to come up in every customer EHS review,
and expect the customer's facility electrical inspector and their insurer to ask
about NRTL listing — which is the practical hook that makes UL 3300 matter.

---

## 5. Non-deterministic control: the honest answer

**ISO 13849's entire framework — probability of dangerous failure per hour, mean
time to dangerous failure, diagnostic coverage — assumes you can characterise the
failure rates of components. There is no accepted method for assigning a
dangerous-failure rate to a neural policy.**

ISO 10218:2025 added functional safety requirements and cybersecurity provisions
but its foundational assumption is deterministic control and static stability.
ISO 25785-1 is unfinished. The EU Machinery Regulation names the problem —
*"data dependency, opacity, autonomy and connectivity"* — without solving it.

> **Nobody is certifying a neural network to PLd, and the field's answer is
> unanimous: don't try.** The universal pattern is that the safety layer is
> deterministic, certified and *independent of the machine learning*, and the
> policy is treated as an **untrusted input, envelope-limited by something that
> can be characterised.**
>
> **The standards will not rescue you. Build accordingly.**

---

## 6. What a supervised pilot actually requires

A supervised pilot is **not exempt from most of it — but it is exempt from the
expensive parts.**

### The EU test: is it a supply, or a test?

Sourced positions: CE marking is *"not required for a prototype during R&D
stage, as it is not posed to the market yet, but it **is** applicable if free
samples are sent to customers or a product is rented"*; and *"putting a
prototype on the market or providing it to a customer for anything other than
strictly supervised product testing can trigger CE requirements."*

Read together, the operative test is clear. **A pilot survives without CE
marking only if it is genuinely a supervised test, not a supply:**

- The contract is a **test/evaluation agreement**, not a sale, lease or rental
- **You retain ownership and control**; the customer does not operate it unattended
- **Your** trained operator is present and in control, with the e-stop in reach
- Defined schedule, defined area, defined evaluation purpose
- You collect and act on findings — it looks like research and development
  because it *is*

The moment customer staff operate it unsupervised, or money changes hands for the
robot's output as a product, or you leave it there over a weekend, you have
arguably placed it on the market.

⚠️ **Get a compliance lawyer to sign off on the contract wording. This is a
€1–3k spend that de-risks the entire pilot.**

**In the US** there is no pre-market approval for a robot. You can run a
supervised pilot on risk assessment plus OSHA-compliant work practices. **The
gate is not a regulator; the gate is the customer's EHS department and their
insurer.**

### The minimum package

| Item | Effort / cost |
|---|---|
| **ISO 12100 risk assessment**, documented and signed | 1–3 weeks internal, or ~€2–5k consultant |
| **Hard-wired e-stop, PLd Category 3**, dual-channel, tested | ~€500–1,500 in components (safety relay + rated e-stops) |
| Safety-rated speed and force limiting **below** the policy | engineering time |
| Written safe-work procedure, defined exclusion zone (barriers, tape, floor markings) | days |
| **Trained operator present at all times**, with training records | — |
| **Customer site agreement**: test agreement, indemnity, access control, incident reporting | legal review **€1–3k** |
| General liability + product liability insurance | see below |
| Incident log and escalation path, agreed with customer EHS | days |
| Lockout/tagout procedure for maintenance (US: 1910.147) | days |

**Legitimately deferrable for a supervised pilot:** CE marking and the full
technical file, Notified Body involvement, UL 3300 listing, EU AI Act conformity
work, and full ISO 13849 validation of the whole system — though you *should*
do it for the e-stop chain.

---

## 7. Costs and timelines

⚠️ **All figures in this section come from compliance-consultancy marketing
pages, not from Notified Bodies. Order-of-magnitude only.**

| Component | Range |
|---|---|
| Machinery CE marking, total, self-assessed | **€1,500–5,000+** |
| Compliance consultant | €1,000–5,000+ |
| Accredited lab testing | €500–5,000+ |
| **Notified Body fees, where required** | **€5,000–15,000+** |
| Technical documentation | €500–3,000+ |

Timelines: machinery requiring Notified Body assessment **3–6 months** (one
source says 4–8); Notified Body involvement alone adds **4–12 weeks** depending
on their queue; simple electronics without a Notified Body, 4–6 weeks.

The one line from the source material worth quoting, because it is the true
finding:

> *"The larger cost is usually the rework and delay when documentation is
> incomplete, not the testing invoice."*

⚠️ **Assessment:** the quoted €5–15k Notified Body fee is realistic for
straightforward machinery and **optimistic for a novel ML-driven mobile
manipulator**, where the assessor has no precedent and will bill for the learning
curve. Budget more — or, better, use §1 to avoid needing one.

### Insurance

⚠️ **Weak sourcing — broker marketing pages, not quotes.** General liability
limits for robotics are typically **$1M to $5M or higher**. One aggregator
advises 2–5% of total robot investment annually for comprehensive coverage
(low-quality source; flagged).

⚠️ *Standard practice, not sourced:* the customer will require a certificate of
insurance naming them before you set foot on site, usually **$1M per occurrence
/ $2M aggregate** minimum, sometimes $5M for industrial sites. **Product
liability is the hard one** for a pre-revenue company shipping ML-driven motion —
expect underwriter questions about the e-stop architecture and whether ML
controls safety functions, **which is another reason the §1 separation pays for
itself**. Engage a broker **8–12 weeks before** the pilot, not two. Specialty
lines are slow.

---

## 8. The EU AI Act timeline, including the 2026 delay

Original timeline: in force 2024-08-01; prohibitions and AI literacy
2025-02-02; general-purpose AI obligations, notified bodies, governance and
penalties 2025-08-02; high-risk (except Art. 6(1)) 2026-08-02; **Art. 6(1)
embedded high-risk 2027-08-02**.

The **Digital Omnibus** changed this in 2026, and is now settled, citable law:
**Regulation (EU) 2026/1744** (CELEX 32026R1744), signed at Strasbourg
2026-07-08, **in force 2026-07-27** — confirmed both by the amended AI Act
Art. 113 text and by the Commission's AI policy page. (This resolves the
warning the 2026-08-08 edition of this doc carried here, which found the
in-force date unsettled: it settled within the week. Regulatory claims decay
fastest.)

Confirmed dates from the enacted Art. 113:

- Annex III standalone high-risk pushed **2026-08-02 → 2027-12-02**
- **Annex I embedded high-risk — AI inside machinery, which is this case —
  pushed to 2028-08-02** (the same date the MR Art. 8 delegated acts of §1
  apply by — the two regimes now land together, by design)

The Omnibus also obliges the Commission to publish burden-minimisation
guidelines (Arts. 8(2)/9(10)/17(3)) by **2027-08-01** and post-market-monitoring
guidance with template by **2027-09-02**. As of 2026-08-15 the AI Office has
published guidelines on prohibited practices, the AI-system definition, GPAI
scope, transparency (2026-07-20), high-risk providers/deployers, and an
MDR/IVDR interplay paper — but **no machinery–AI-Act interplay guidance yet**;
"interplay with other EU legislation" sits on its in-progress list.

Either way, the Machinery Regulation's **2027-01-20** date arrives first, and §1
is what matters.

---

## 9. Sources

EU Machinery Regulation (EU) 2023/1230, OJ L 165, 2023-06-29
<https://eur-lex.europa.eu/legal-content/EN/LSU/?uri=CELEX:32023R1230> ·
Regulation (EU) 2026/1744 (Digital Omnibus, in force 2026-07-27)
<https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32026R1744> ·
European Commission AI Act Service Desk, Art. 6(1) guidance ·
Commission AI policy page
<https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai> ·
"Horizontal RfUs in force MR — Status May 2026" (RfU CNB/M/00.514 Rev 02 and
00.518), linked from
<https://single-market-economy.ec.europa.eu/sectors/mechanical-engineering/machinery_en> ·
ISO 10218-1:2025 / -2:2025 · ANSI/A3 R15.06-2025 · ANSI/A3 R15.08 Parts 1–3 ·
ISO 3691-4:2023 · ISO 13482 / ISO/FDIS 13482 · ISO 25785-1 (in development) ·
ANSI/CAN/UL 3300:2024 · ISO 13849-1:2023 · IEC 60204-1 · ISO 12100 ·
OSHA robotics standards <https://www.osha.gov/robotics/standards> · 29 CFR 1910 ·
CE-marking cost and prototype-exemption positions: TESTiLABS, tcfcert and
comparable compliance consultancies (marketing pages — treat as indicative)
