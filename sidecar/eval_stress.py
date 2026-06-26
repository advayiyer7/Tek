"""Comprehensive retrieval stress eval: a deliberately adversarial benchmark.

Where eval_retrieval.py is the saturated correctness gate (19/19 by design),
this harness is built to make the pipeline drop points, so the headline
statistic is honest:

- Confusable clusters: many files sharing vocabulary (8 sprint notes, 4 roman
  pastas, 5 apartment docs, 4 homelab docs, ...) where only one answers.
- Near-duplicate versions (contract v1/v2, lease 2023/2024).
- Long-document needles: facts buried deep inside multi-thousand-word files.
- Typo'd, vague, and zero-keyword-overlap paraphrase queries.
- Adversarial negatives: topically adjacent questions with no answer in the
  corpus (the no-answer floor must reject them).
- Hostile files: null bytes, single 100KB line, CJK/emoji/RTL, deep nesting.
- Distractor mass: seeded pseudo-realistic filler notes; --scale grows the
  index past 20k chunks so the IVF-PQ ANN path (not brute force) is measured.

Metrics, per category and overall: top-1 accuracy, recall@5, MRR@10,
negative rejection rate, two-hop recall@8, query latency p50/p95,
indexing throughput.

Run:  .venv/Scripts/python eval_stress.py [--scale] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval_retrieval import CORPUS as CORE_CORPUS
from eval_retrieval import PROBES as LEGACY_PROBES
from tek.config import Config
from tek.embed import FastEmbedEmbedder
from tek.indexer import Indexer
from tek.rag import retrieve
from tek.rerank import Reranker
from tek.scanner import scan_folders
from tek.store import Store

# --------------------------------------------------------------------------
# Corpus: hand-authored adversarial clusters
# --------------------------------------------------------------------------

CLUSTER_CORPUS: dict[str, str] = {
    # -- Sprint meeting notes: 8 files sharing standup/retro vocabulary ----
    "work/meetings/sprint-50.md": (
        "# Sprint 50 notes\n\nStandup recap: deploy pipeline still slow, retro went long. "
        "Decision: migrate CI off Jenkins to GitHub Actions; Devon will write the workflow files. "
        "Priya owns the flaky checkout test until it is green for two weeks. "
        "Payments and search both quiet this sprint."
    ),
    "work/meetings/sprint-51.md": (
        "# Sprint 51 notes\n\nStandup recap: retro focused on infra debt. "
        "Decision: the Redis upgrade is postponed to Q3 because the client library needs an audit first. "
        "The new logo shipped to staging behind a flag. Deploy cadence stays weekly."
    ),
    "work/meetings/sprint-52.md": (
        "# Sprint 52 notes\n\nStandup recap: analytics service kickoff. "
        "Decision: Postgres 16 over MySQL for the analytics service, mainly for partitioned tables. "
        "We signed two contractors for the data migration. Retro: fewer meetings, more pairing."
    ),
    "work/meetings/sprint-53.md": (
        "# Sprint 53 notes\n\nStandup recap: payments incident follow-up. "
        "The payments retry storm was root-caused to clock skew between the worker pool and the queue. "
        "Decision: on-call rotation moves from biweekly to weekly. Retro was cancelled."
    ),
    "work/meetings/sprint-54.md": (
        "# Sprint 54 notes\n\nStandup recap: mobile beta slipped two weeks for app-store review. "
        "Decision: LaunchDarkly approved as the feature flag vendor after the build-vs-buy spike. "
        "Search team demoed query suggestions in the retro."
    ),
    "work/meetings/sprint-55.md": (
        "# Sprint 55 notes\n\nStandup recap: deprecation planning. "
        "Decision: the v1 REST API will be turned off on November 30; clients get a migration guide and two reminder emails. "
        "Retro highlighted that deploy windows keep colliding with on-call handoff."
    ),
    "work/meetings/sprint-56.md": (
        "# Sprint 56 notes\n\nStandup recap: performance week. "
        "Decision: the search latency SLO is set at p95 350ms measured at the gateway. "
        "Anna presented the embeddings prototype in the retro; payments quiet."
    ),
    "work/meetings/sprint-57.md": (
        "# Sprint 57 notes\n\nStandup recap: incident review for the March 4 outage. "
        "Root cause: an expired TLS certificate on the edge proxy. "
        "Action: automate renewal with cert-manager so no cert is ever hand-rotated again. "
        "Retro: postmortems move to a blameless template."
    ),
    # -- Roman pastas: shared guanciale/pecorino vocabulary -----------------
    "recipes/cacio_e_pepe.md": (
        "# Cacio e pepe\n\nJust three things: pecorino romano, black pepper, pasta water. "
        "Toast the pepper, emulsify the cheese off the heat with starchy water. "
        "No guanciale, no eggs, no tomato — the pepper is the whole show."
    ),
    "recipes/amatriciana.md": (
        "# Amatriciana\n\nThe red one of the roman pastas: guanciale rendered crisp, "
        "tomatoes (San Marzano), a little chili, finished with pecorino romano. "
        "Traditionally bucatini. No eggs, no cream, no onion in the strict version."
    ),
    "recipes/gricia.md": (
        "# Gricia\n\nGricia is essentially carbonara without the egg: guanciale, "
        "pecorino romano, black pepper, pasta water. Also amatriciana without the tomato. "
        "The oldest of the four roman pastas."
    ),
    "recipes/ragu_bolognese.md": (
        "# Ragù bolognese\n\nSoffritto, beef and pork, a little milk, white wine (not red), "
        "just a spoon of tomato paste — it is a meat sauce, not a tomato sauce. "
        "Simmer three hours minimum. Serve with tagliatelle, never spaghetti."
    ),
    # -- Apartment: lease versions + adjacent docs ---------------------------
    "apartment/lease_2024.md": (
        "# Lease agreement — ACTIVE\n\nThis is the current, active lease for unit 4B at 88 Alder Street, "
        "signed June 1 2024 with landlord Marta Chen. Monthly rent: $1,800 due on the 1st. "
        "Term ends May 31 2026. Pets: no cats or dogs without the landlord's written consent. "
        "Subletting prohibited. Tenant handles utilities except water."
    ),
    "apartment/lease_2023_expired.md": (
        "# Lease agreement — EXPIRED 2023\n\nThe old lease for unit 4B, in force during 2023 only. "
        "Monthly rent was $1,650 due on the 1st. Same landlord, Marta Chen. "
        "Superseded by the June 2024 lease; kept for records."
    ),
    "apartment/renewal_offer_2026.md": (
        "# Renewal offer (2026)\n\nMarta's renewal proposal for unit 4B starting June 2026: "
        "rent goes from $1,800 to $1,895, a 5.3% increase. Twelve-month term. "
        "I must respond by April 30 or the unit goes on the market."
    ),
    "apartment/deposit_receipt.md": (
        "# Security deposit receipt\n\nPaid $2,700 security deposit for unit 4B on June 3 2024 "
        "(1.5x one month). Refundable at move-out less documented damages, returned within 21 days. "
        "Receipt #DR-1187 signed by Marta Chen."
    ),
    "apartment/movein_inspection.md": (
        "# Move-in inspection — unit 4B\n\nNoted at move-in, June 2024: bathroom grout cracked along the tub, "
        "scuff marks in the hallway, the dishwasher's lower rack is rusty, balcony door sticks. "
        "Photos attached to email thread. Landlord acknowledged all four items."
    ),
    # -- Homelab: overlaps wifi.txt's 192.168.1.x space ---------------------
    "homelab/proxmox.md": (
        "# Proxmox host\n\nThe Proxmox box is at 192.168.1.50, 64GB RAM, hosts the virtual machines: "
        "jellyfin media server on 192.168.1.51 (holds all the movies), home assistant on 192.168.1.52. "
        "Backups of VM snapshots go to the NAS weekly."
    ),
    "homelab/pihole.md": (
        "# Pi-hole\n\nDNS-level ad blocking for every device at home runs on the Pi-hole at 192.168.1.53. "
        "Upstream resolver is 1.1.1.1. It blocks roughly 18% of queries. "
        "If a site breaks, whitelist it in the admin panel rather than disabling blocking."
    ),
    "homelab/vpn.md": (
        "# WireGuard VPN\n\nWireGuard listens on UDP port 51820, endpoint home.ddns.example.net. "
        "Peer configs exist for the phone and the laptop. "
        "Keepalive 25s because the router NATs aggressively. Keys live in the password manager."
    ),
    "homelab/backups.md": (
        "# Backup scheme\n\nrestic pushes encrypted snapshots to Backblaze B2 nightly at 2am. "
        "Retention: 7 daily, 8 weekly, 12 monthly snapshots. "
        "Restore drill every quarter: pull a random file and verify the hash."
    ),
    # -- Cars ---------------------------------------------------------------
    "cars/civic_log.md": (
        "# Civic maintenance log\n\n2019 Honda Civic. Last oil change at 62,300 miles on May 10 "
        "(full synthetic). Next service due at 67,000 miles. "
        "Winter tires are in storage unit 14 at StorQuest, mounted on steel rims. "
        "Cabin filter replaced same visit."
    ),
    "cars/rav4_log.md": (
        "# RAV4 maintenance log\n\n2021 Toyota RAV4 (wife's car). Last oil change at 48,150 miles. "
        "Next service due at 53,000 miles. Runs all-season tires year-round. "
        "Rear wiper still streaks after replacement — try the OEM blade next."
    ),
    "cars/auto_insurance.md": (
        "# Auto insurance\n\nGoodHands policy POL-88421 covers both cars. Premium $128 per month, "
        "deductible $500 collision / $250 comprehensive. Renews October 12. "
        "Roadside assistance included; glass claims don't affect the premium."
    ),
    # -- Contract versions ----------------------------------------------------
    "contracts/freelance_v1.md": (
        "# Freelance contract v1 (January 2025) — superseded\n\nOriginal terms signed in January: "
        "$95/hour, payment net-45, two revision rounds included per deliverable. "
        "No termination clauses beyond 14-day notice. Replaced by v2 in April."
    ),
    "contracts/freelance_v2.md": (
        "# Freelance contract v2 (April 2025) — CURRENT\n\nRenegotiated terms, supersedes v1: "
        "$105/hour, payment net-30, three revision rounds included per deliverable. "
        "New in v2: a 25% kill fee if a project is cancelled after kickoff (v1 had no kill fee). "
        "14-day notice unchanged."
    ),
    # -- Contacts -------------------------------------------------------------
    "contacts/dr_patel.md": (
        "# Dr. Patel — dermatologist\n\nAppointment August 14 at 2:30pm, office on Marine Drive, suite 410. "
        "Ask about the tretinoin refill and the mole on the left shoulder. "
        "Bring the referral letter; parking validated in the building."
    ),
    "contacts/accountant.md": (
        "# Meera Shah, CPA\n\nMy accountant: meera@shahcpa.example, (555) 014-2287. "
        "Files my returns quarterly. She wants all 1099s sent to her by January 20, "
        "and receipts for any deduction over $75. Books a review call every March."
    ),
    "contacts/trainer.md": (
        "# Coach Sam — personal training\n\nSessions Tuesdays 7am at the east-side gym. "
        "Current block focuses on hip mobility and single-leg strength. "
        "Sam's rule: if sleep was under 6 hours, swap the heavy session for technique work."
    ),
    # -- Money docs -----------------------------------------------------------
    "money/invoice_acme_0231.md": (
        "# Invoice INV-0231 — Acme Corp\n\nIssued June 12 for the website accessibility audit. "
        "Amount due: $4,800, payment due July 10, net-30 from issue. "
        "Contact: ap@acme.example. Late fee 1.5%/month after due date."
    ),
    "money/invoice_globex_0232.md": (
        "# Invoice INV-0232 — Globex\n\nIssued June 24 for the logo redesign package. "
        "Amount due: $2,200, payment due July 24. "
        "Globex pays by check (allow a week). PO number GLX-2025-118 must appear on the invoice."
    ),
    "money/receipt_macbook.md": (
        "# Receipt — MacBook Air\n\nBought a MacBook Air M3 13-inch for $1,899 on March 2 at the Apple Store. "
        "Serial C02XK1ABCDE. AppleCare+ declined. "
        "Business purchase — give this receipt to Meera for the deduction."
    ),
    "money/warranty_monitor.md": (
        "# Monitor warranty\n\nDell U2723QE 27-inch, service tag 7TQR3X2. "
        "Three-year advanced-exchange warranty, covered until June 2027. "
        "Dead-pixel policy: 1 bright or 5 dark pixels qualify for replacement."
    ),
}

# -- Hostile edge-case files (indexer must survive; some carry needles) ------

EDGE_TEXT_FILES: dict[str, str] = {
    "edge/unicode_notes.md": (
        "# 会議メモ / ملاحظات / Заметки 📝\n\n"
        "日本語のテキストが含まれています。これは多言語ストレステストです。\n\n"
        "هذا نص عربي من اليمين إلى اليسار للاختبار.\n\n"
        "Ёмкая строка на русском языке. Emoji soup: 🚀🔥💾🧪🌍✨\n\n"
        "One English fact hides here: the conference wifi password is falcon-velvet-9012, "
        "taped under the badge desk.\n\n"
        "Mixed: naïve café résumé Zürich Œuvre ﬁligree."
    ),
    "edge/nested/a/b/c/d/e/deep_note.md": (
        "# Deep note\n\nBuried five folders down on purpose. "
        "The storage unit access code is 4417# — punch it at the StorQuest gate keypad, "
        "then lift the latch while the light is green."
    ),
    "edge/UPPER CASE & (parens) note.MD": (
        "# Weird filename test\n\nThis file has spaces, an ampersand, parentheses and an upper-case "
        "extension. Fact: the spare house key is taped inside the electrical panel, not under the mat."
    ),
}


def build_long_docs(rng: random.Random) -> dict[str, str]:
    """Three long documents with needles buried mid-stream."""
    docs: dict[str, str] = {}

    moods = ["calm", "scattered", "focused", "restless", "tired but fine", "weirdly productive"]
    topics = [
        "cleaned the garage shelves and found old cables",
        "long walk by the river, podcasts the whole way",
        "meal-prepped lentils and roasted vegetables for the week",
        "fixed a squeaky door hinge with the wrong tool, twice",
        "read two chapters and fell asleep on the couch",
        "called my parents, dad is rebuilding the fence again",
        "tried a new espresso ratio, slightly too bitter",
        "sorted the photo library for an hour, barely a dent",
        "watched a documentary about deep sea vents",
        "weeded the planter boxes before it got hot",
    ]
    entries = []
    for i in range(1, 61):
        mood = rng.choice(moods)
        body = " ".join(rng.sample(topics, 2))
        entries.append(f"## Day {i}\n\nFelt {mood} today. {body}. Slept about {rng.randint(6, 9)} hours.")
    entries[36] += (
        " Also finally did the passport paperwork — the passport renewal confirmation number "
        "is PR-2210394, keep it until the new one arrives."
    )
    entries[21] += " Lent Marcus my copy of Dune at lunch; he promises it back by the end of the month."
    docs["journal/journal_2025.md"] = "# Journal 2025\n\n" + "\n\n".join(entries)

    speakers = ["Ana", "Devon", "Priya", "Marcus", "Lee"]
    lines = []
    fillers = [
        "I think we should park that for now and revisit after the break.",
        "Can everyone see the dashboard? Okay, moving on.",
        "That depends on the migration timeline we discussed earlier.",
        "Let's not redesign this in the meeting, take it offline.",
        "Agreed, but someone has to own the rollout checklist.",
        "The numbers from last quarter don't really support that.",
        "We keep saying that every quarter and it never happens.",
        "Good point — add it to the risks section.",
        "I'll have a draft by Friday if nothing blows up.",
        "Can we get the contractors access before Monday?",
    ]
    for i in range(220):
        lines.append(f"{rng.choice(speakers)}: {rng.choice(fillers)}")
    lines[131] = (
        "Ana: Final decision then — the public API rate limit will be 1,200 requests "
        "per minute per key, enforced at the gateway. Everyone agreed."
    )
    lines[172] = "Lee: And the offsite is confirmed for Lisbon in September, flights book next week."
    lines[58] = "Devon: On headcount, we are approved to hire two backend engineers in Q3, no more."
    docs["work/q1_planning_transcript.md"] = "# Q1 planning — full transcript\n\n" + "\n".join(lines)

    concepts = [
        "gradient descent converges when the learning rate respects the Lipschitz constant",
        "regularization trades variance for bias; L1 induces sparsity, L2 shrinks smoothly",
        "cross-validation estimates generalization; never tune on the test fold",
        "decision trees overfit unpruned; ensembles average away the variance",
        "kernel methods implicitly map to high-dimensional feature spaces",
        "backpropagation is reverse-mode autodiff applied to the loss graph",
        "batch norm stabilizes the distribution of intermediate activations",
        "attention weighs value vectors by query-key similarity",
        "dropout approximates an ensemble of subnetworks at train time",
        "early stopping is regularization measured against the validation curve",
    ]
    paras = []
    for i in range(1, 41):
        pts = rng.sample(concepts, 3)
        paras.append(
            f"## Lecture note block {i}\n\nKey points: {pts[0]}. Related: {pts[1]}. "
            f"Also covered: {pts[2]}. Worked examples in the problem set."
        )
    paras[24] += (
        " ADMIN NOTE: the midterm is on October 17 and covers lectures 1 through 9 only; "
        "one cheat sheet allowed, both sides."
    )
    docs["school/ml_course_notes.md"] = "# Machine learning course notes\n\n" + "\n\n".join(paras)
    return docs


# -- Distractor filler: pseudo-realistic notes sharing cluster vocabulary ----

FILLER_TOPICS: dict[str, list[str]] = {
    "cooking": [
        "Tried a new pasta shape; the sauce clung better than expected.",
        "Note to self: salt the water like the sea, every recipe says it for a reason.",
        "The cast iron needs reseasoning after the tomato braise.",
        "Overnight dough in the fridge develops more flavor than same-day.",
        "Cheese before plating, never while the pan is screaming hot.",
        "Stock from scraps simmered all afternoon; freezer is stocked.",
    ],
    "fitness": [
        "Easy run day, kept the heart rate low and it felt slow but right.",
        "Grip strength is the quiet bottleneck in every pull exercise.",
        "Stretching before bed seems to help more than in the morning.",
        "Progressive overload only counts if the form holds.",
        "Rest days are training days for connective tissue.",
        "Tracked protein for a week; breakfast is where it falls short.",
    ],
    "money": [
        "Reviewed subscriptions; cancelled two streaming services nobody watched.",
        "Index funds are boring on purpose — that is the entire point.",
        "Moved the emergency fund to a higher-yield savings account.",
        "Receipts pile up fast; scanning them weekly keeps the shoebox empty.",
        "The grocery budget holds if the list is written before the store.",
        "Annual fees sneak in quietly; calendar reminders catch them.",
    ],
    "homelab_ish": [
        "Cable management under the desk finally done with velcro ties.",
        "The old laptop became a test box; runs quiet with the lid closed.",
        "Router firmware updated; the changelog was three lines long.",
        "Labelled every power brick; future me says thank you.",
        "Network switch hums slightly; replaced the fan with a quieter one.",
        "Documented the rack layout in a diagram nobody else will read.",
    ],
    "work_ish": [
        "Meetings that could be emails were emails today; a good day.",
        "Wrote the design doc first and the code went twice as fast.",
        "Code review backlog cleared before lunch for once.",
        "The roadmap shifted again; priorities are a moving target.",
        "Paired with a teammate on a gnarly bug; two heads, one fix.",
        "Inbox zero achieved at 4pm; lasted eleven minutes.",
    ],
    "garden_ish": [
        "Deadheaded the planters; the second bloom is always the better one.",
        "Compost bin turned; the steam in the morning means it is working.",
        "The north bed gets less light than the seed packet wants.",
        "Slugs found the seedlings; copper tape goes down tomorrow.",
        "Rain barrel filled overnight; watering is free this week.",
        "Repotted the rosemary; it was rootbound something fierce.",
    ],
    "travel_ish": [
        "Packing cubes changed the carry-on game completely.",
        "Museum mornings, café afternoons — the only itinerary that works.",
        "Booked the window seat on the left side for the mountain view.",
        "Offline maps downloaded before the airport this time.",
        "The best meal of the trip came from a stall with no name.",
        "Travel insurance reads like fiction until the one time it does not.",
    ],
    "reading_ish": [
        "Two bookmarks in two books; one always wins by Friday.",
        "Library hold came in the same week as the busy sprint, naturally.",
        "Margins full of pencil notes; future re-reads will be a conversation.",
        "Short stories before bed beat doomscrolling every single time.",
        "The sequel is slower but the world-building pays the rent.",
        "Re-read a childhood favorite; it grew up alongside me.",
    ],
}


def build_filler(count: int, paragraphs_per_file: tuple[int, int], rng: random.Random) -> dict[str, str]:
    """Deterministic distractor notes. Generic by construction: they share
    vocabulary with the clusters but never state any probed fact."""
    topics = list(FILLER_TOPICS)
    out: dict[str, str] = {}
    for i in range(count):
        topic = topics[i % len(topics)]
        n_paras = rng.randint(*paragraphs_per_file)
        paras = []
        for _ in range(n_paras):
            sentences = rng.sample(FILLER_TOPICS[topic], k=min(4, len(FILLER_TOPICS[topic])))
            paras.append(" ".join(sentences))
        out[f"archive/{topic}/note_{i:04d}.md"] = f"# {topic} note {i}\n\n" + "\n\n".join(paras)
    return out


# --------------------------------------------------------------------------
# Probes. accept = any of these files counts as a correct top-1.
# --------------------------------------------------------------------------

PROBES: list[tuple[str, tuple[str, ...], str]] = [
    # ---- legacy (the original 19, for continuity) — added in main() ----
    # ---- direct facts on new clusters ----
    ("who is my landlord", ("apartment/lease_2024.md", "apartment/lease_2023_expired.md", "apartment/deposit_receipt.md", "apartment/renewal_offer_2026.md"), "direct"),
    ("what unit number is my apartment", ("apartment/lease_2024.md", "apartment/lease_2023_expired.md", "apartment/deposit_receipt.md", "apartment/movein_inspection.md", "apartment/renewal_offer_2026.md"), "direct"),
    ("how much was the security deposit", ("apartment/deposit_receipt.md",), "direct"),
    ("what port does wireguard listen on", ("homelab/vpn.md",), "direct"),
    ("what is the upstream DNS resolver", ("homelab/pihole.md",), "direct"),
    ("how much RAM does the proxmox host have", ("homelab/proxmox.md",), "direct"),
    ("what is my auto insurance premium", ("cars/auto_insurance.md",), "direct"),
    ("when is my dermatologist appointment", ("contacts/dr_patel.md",), "direct"),
    ("what is my accountant's email", ("contacts/accountant.md",), "direct"),
    ("what day are my training sessions", ("contacts/trainer.md",), "direct"),
    ("how much was the macbook", ("money/receipt_macbook.md",), "direct"),
    ("what does bolognese get instead of red wine", ("recipes/ragu_bolognese.md",), "direct"),
    ("how long do I simmer bolognese", ("recipes/ragu_bolognese.md",), "direct"),
    ("what keeps the wireguard connection alive behind NAT", ("homelab/vpn.md",), "direct"),
    ("how often is the backup restore drill", ("homelab/backups.md",), "direct"),
    # ---- exact-keyword (BM25 must carry) ----
    ("POL-88421", ("cars/auto_insurance.md",), "keyword"),
    ("INV-0231", ("money/invoice_acme_0231.md",), "keyword"),
    ("C02XK1ABCDE", ("money/receipt_macbook.md",), "keyword"),
    ("7TQR3X2", ("money/warranty_monitor.md",), "keyword"),
    ("51820", ("homelab/vpn.md",), "keyword"),
    ("192.168.1.53", ("homelab/pihole.md",), "keyword"),
    ("192.168.1.52", ("homelab/proxmox.md",), "keyword"),
    ("PR-2210394", ("journal/journal_2025.md",), "keyword"),
    ("cert-manager", ("work/meetings/sprint-57.md",), "keyword"),
    ("LaunchDarkly", ("work/meetings/sprint-54.md",), "keyword"),
    ("StorQuest gate code", ("edge/nested/a/b/c/d/e/deep_note.md", "cars/civic_log.md"), "keyword"),
    ("Backblaze B2", ("homelab/backups.md",), "keyword"),
    ("GLX-2025-118", ("money/invoice_globex_0232.md",), "keyword"),
    ("DR-1187", ("apartment/deposit_receipt.md",), "keyword"),
    ("falcon-velvet-9012", ("edge/unicode_notes.md",), "keyword"),
    ("U2723QE", ("money/warranty_monitor.md",), "keyword"),
    # ---- zero-keyword-overlap paraphrases ----
    ("am I allowed to get a cat", ("apartment/lease_2024.md",), "paraphrase"),
    ("how much more will I pay if I stay another year", ("apartment/renewal_offer_2026.md",), "paraphrase"),
    ("money I get back when I move out", ("apartment/deposit_receipt.md",), "paraphrase"),
    ("how do I stop ads on every device at home", ("homelab/pihole.md",), "paraphrase"),
    ("recovering work after a bad git command", ("tech/git_tips.md",), "paraphrase"),
    ("what should I drink less of to sleep better", ("health/sleep_notes.txt",), "paraphrase"),
    ("how much do I charge per hour now", ("contracts/freelance_v2.md",), "paraphrase"),
    ("who helps me file my tax returns", ("contacts/accountant.md", "finance/tax_notes_2025.md"), "paraphrase"),
    ("the cream I use on my skin", ("contacts/dr_patel.md",), "paraphrase"),
    ("which computer holds all the movies", ("homelab/proxmox.md",), "paraphrase"),
    ("the roman pasta that is just cheese and pepper", ("recipes/cacio_e_pepe.md",), "paraphrase"),
    ("what happens if a client cancels a project after we start", ("contracts/freelance_v2.md",), "paraphrase"),
    ("what was wrong with the kitchen appliance when I got the keys", ("apartment/movein_inspection.md",), "paraphrase"),
    ("how fast do searches need to be now", ("work/meetings/sprint-56.md", "work/standup_notes.md"), "paraphrase"),
    ("where is the spare key hidden", ("edge/UPPER CASE & (parens) note.MD",), "paraphrase"),
    ("what did the doctor need to check on my back", ("contacts/dr_patel.md",), "paraphrase"),
    ("how do I get into my storage space", ("edge/nested/a/b/c/d/e/deep_note.md",), "paraphrase"),
    ("which invoice needs a purchase order number on it", ("money/invoice_globex_0232.md",), "paraphrase"),
    # ---- confusable clusters: distractors share vocabulary ----
    ("who owns the flaky checkout test", ("work/meetings/sprint-50.md",), "confusable"),
    ("when did we decide to leave Jenkins", ("work/meetings/sprint-50.md",), "confusable"),
    ("when was the Redis upgrade pushed to", ("work/meetings/sprint-51.md",), "confusable"),
    ("which database did we pick for the analytics service", ("work/meetings/sprint-52.md",), "confusable"),
    ("what caused the payments retry storm", ("work/meetings/sprint-53.md",), "confusable"),
    ("which feature flag vendor did we approve", ("work/meetings/sprint-54.md",), "confusable"),
    ("when is the v1 REST API being shut down", ("work/meetings/sprint-55.md",), "confusable"),
    ("what p95 latency target did we set for search", ("work/meetings/sprint-56.md",), "confusable"),
    ("why did the site go down on March 4", ("work/meetings/sprint-57.md",), "confusable"),
    ("what was blocking the auth refactor in sprint 42", ("work/standup_notes.md",), "confusable"),
    ("what was the search latency regression traced to", ("work/standup_notes.md",), "confusable"),
    ("which pasta has no eggs and no tomato", ("recipes/gricia.md", "recipes/cacio_e_pepe.md"), "confusable"),
    ("which roman pasta is the red one", ("recipes/amatriciana.md",), "confusable"),
    ("does amatriciana have onion", ("recipes/amatriciana.md",), "confusable"),
    ("what was my rent under the old lease", ("apartment/lease_2023_expired.md",), "confusable"),
    ("what is my current monthly rent", ("apartment/lease_2024.md", "finance/budget.txt"), "confusable"),
    ("how big is the proposed rent increase", ("apartment/renewal_offer_2026.md",), "confusable"),
    ("what did the inspection note about the bathroom", ("apartment/movein_inspection.md",), "confusable"),
    ("which IP is the ad blocker on", ("homelab/pihole.md",), "confusable"),
    ("which address does the NAS reserve", ("notes/wifi.txt",), "confusable"),
    ("what time do nightly backups run", ("homelab/backups.md",), "confusable"),
    ("what mileage was the civic's last oil change", ("cars/civic_log.md",), "confusable"),
    ("when is the rav4 due for service", ("cars/rav4_log.md",), "confusable"),
    ("what is my collision deductible", ("cars/auto_insurance.md",), "confusable"),
    ("how much does Acme owe me", ("money/invoice_acme_0231.md",), "confusable"),
    ("when is the Globex invoice due", ("money/invoice_globex_0232.md",), "confusable"),
    ("how long is the monitor covered", ("money/warranty_monitor.md",), "confusable"),
    ("which car runs all-season tires", ("cars/rav4_log.md",), "confusable"),
    # ---- version disambiguation ----
    ("payment terms in the latest contract", ("contracts/freelance_v2.md",), "version"),
    ("payment terms in the original contract", ("contracts/freelance_v1.md",), "version"),
    ("did the old contract have a kill fee", ("contracts/freelance_v2.md", "contracts/freelance_v1.md"), "version"),
    ("my hourly rate before the renegotiation", ("contracts/freelance_v1.md",), "version"),
    ("how many revision rounds do clients get now", ("contracts/freelance_v2.md",), "version"),
    ("which lease is active right now", ("apartment/lease_2024.md",), "version"),
    # ---- temporal anchors ----
    ("when does my lease end", ("apartment/lease_2024.md",), "temporal"),
    ("what must I respond to by April 30", ("apartment/renewal_offer_2026.md",), "temporal"),
    ("what does Meera need by January 20", ("contacts/accountant.md",), "temporal"),
    ("when does the auto insurance renew", ("cars/auto_insurance.md",), "temporal"),
    ("what is due at 67,000 miles", ("cars/civic_log.md",), "temporal"),
    ("until when is the monitor under warranty", ("money/warranty_monitor.md",), "temporal"),
    ("what happened on March 4", ("work/meetings/sprint-57.md",), "temporal"),
    ("what was bought on March 2", ("money/receipt_macbook.md",), "temporal"),
    # ---- typos ----
    ("sourdoug starter feedng schedual", ("recipes/sourdough.md",), "typo"),
    ("carbonnara creem ok?", ("recipes/carbonara.md",), "typo"),
    ("quaterly estimated taks due dates", ("finance/tax_notes_2025.md",), "typo"),
    ("wherre are teh wintr tires", ("cars/civic_log.md",), "typo"),
    ("dockr prune disk spce", ("tech/docker_cheatsheet.md",), "typo"),
    ("gateron brwon swiches keybord", ("tech/keyboard.txt",), "typo"),
    ("wireguard prot nubmer", ("homelab/vpn.md",), "typo"),
    ("pasport renewal confirmaton number", ("journal/journal_2025.md",), "typo"),
    ("dermatologst appointmnt date", ("contacts/dr_patel.md",), "typo"),
    ("kil fee percentge", ("contracts/freelance_v2.md",), "typo"),
    ("renewl offer respnd by when", ("apartment/renewal_offer_2026.md",), "typo"),
    ("amatricana ingrediants", ("recipes/amatriciana.md",), "typo"),
    ("jellyfn vm ip adress", ("homelab/proxmox.md",), "typo"),
    ("backblze b2 retentn policy", ("homelab/backups.md",), "typo"),
    ("macbok serial numbr", ("money/receipt_macbook.md",), "typo"),
    # ---- vague / underspecified ----
    ("winter tires", ("cars/civic_log.md",), "vague"),
    ("kill fee", ("contracts/freelance_v2.md",), "vague"),
    ("JR pass", ("personal/travel_japan.md",), "vague"),
    ("guest wifi", ("notes/wifi.txt",), "vague"),
    ("deload", ("health/workout.md",), "vague"),
    ("rate limit", ("work/q1_planning_transcript.md", "work/api_design.md"), "vague"),
    ("deposit", ("apartment/deposit_receipt.md",), "vague"),
    ("tretinoin", ("contacts/dr_patel.md",), "vague"),
    ("1099s", ("contacts/accountant.md",), "vague"),
    ("pecorino", ("recipes/cacio_e_pepe.md", "recipes/amatriciana.md", "recipes/gricia.md", "recipes/carbonara.md"), "vague"),
    ("kyoto", ("personal/travel_japan.md",), "vague"),
    ("dead pixels", ("money/warranty_monitor.md",), "vague"),
    # ---- needles in long documents ----
    ("passport renewal confirmation number", ("journal/journal_2025.md",), "needle"),
    ("who did I lend my copy of Dune to", ("journal/journal_2025.md",), "needle"),
    ("what rate limit did we agree on in the planning meeting", ("work/q1_planning_transcript.md",), "needle"),
    ("where is the offsite happening", ("work/q1_planning_transcript.md",), "needle"),
    ("how many backend engineers can we hire", ("work/q1_planning_transcript.md",), "needle"),
    ("when is the machine learning midterm", ("school/ml_course_notes.md",), "needle"),
    ("which lectures does the midterm cover", ("school/ml_course_notes.md",), "needle"),
    ("conference wifi password", ("edge/unicode_notes.md",), "needle"),
    ("storage unit access code", ("edge/nested/a/b/c/d/e/deep_note.md",), "needle"),
    ("am I allowed a cheat sheet in the exam", ("school/ml_course_notes.md",), "needle"),
]

# Two-hop: both files must appear in the top-8 for the query.
MULTIHOP: list[tuple[str, tuple[str, str]]] = [
    ("compare the rent in my old and current lease", ("apartment/lease_2023_expired.md", "apartment/lease_2024.md")),
    ("total amount owed across the Acme and Globex invoices", ("money/invoice_acme_0231.md", "money/invoice_globex_0232.md")),
    ("last oil change mileage for both cars", ("cars/civic_log.md", "cars/rav4_log.md")),
    ("how did my hourly rate change between contract versions", ("contracts/freelance_v1.md", "contracts/freelance_v2.md")),
    ("which box runs the VMs and which one blocks ads", ("homelab/proxmox.md", "homelab/pihole.md")),
    ("what did we decide about Jenkins and about deprecating the v1 API", ("work/meetings/sprint-50.md", "work/meetings/sprint-55.md")),
]

# Negatives: retrieval must return NOTHING. near = topically adjacent traps.
NEGATIVES: list[tuple[str, str]] = [
    ("what is my motorcycle insurance premium", "near"),
    ("what is my dentist's phone number", "near"),
    ("recipe for beef wellington", "near"),
    ("when is my flight to Tokyo", "near"),
    ("my marathon training plan", "near"),
    ("what is the wifi password at the office", "near"),
    ("how much was the electricity bill in March", "near"),
    ("the plumber's quote for the kitchen sink", "near"),
    ("what grade did I get on the ML final exam", "near"),
    ("terms of my mortgage", "near"),
    ("my pilates class schedule", "near"),
    ("what did we decide in sprint 60", "near"),
    ("what is the capital of mongolia", "far"),
    ("transcript of my call with the dentist", "far"),
    ("current price of bitcoin", "far"),
    ("how to replace an iphone screen", "far"),
    ("lyrics to bohemian rhapsody", "far"),
    ("weather forecast for tomorrow", "far"),
    ("best hotels in reykjavik", "far"),
    ("how do whales sleep", "far"),
    ("python global interpreter lock explained", "far"),
    ("steps of the krebs cycle", "far"),
    ("who won the world cup in 2022", "far"),
    ("symptoms of strep throat", "far"),
]


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------

def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(pct / 100 * (len(s) - 1)))))
    return s[idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", action="store_true", help="grow index past 20k chunks (ANN path)")
    ap.add_argument("--no-rerank", action="store_true", help="ablation: fusion ranking only, no CE floor")
    ap.add_argument("--json", default="", help="write results JSON to this path")
    args = ap.parse_args()

    rng = random.Random(7)
    work = Path(tempfile.mkdtemp(prefix="tek-stress-"))
    corpus_dir = work / "corpus"
    data_dir = work / "data"
    results: dict = {"scale": bool(args.scale)}

    try:
        # ---- build corpus on disk ----
        corpus: dict[str, str] = {}
        corpus.update(CORE_CORPUS)
        corpus.update(CLUSTER_CORPUS)
        corpus.update(EDGE_TEXT_FILES)
        corpus.update(build_long_docs(rng))
        if args.scale:
            # Fewer-but-longer files: per-file store writes dominate indexing
            # time, so reach >20k chunks (the ANN threshold) with ~700 files.
            filler = build_filler(1500, (95, 115), rng)
        else:
            filler = build_filler(110, (1, 3), rng)
        corpus.update(filler)

        for rel, content in corpus.items():
            f = corpus_dir / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(content, encoding="utf-8")

        # hostile files outside the text dict
        (corpus_dir / "edge").mkdir(parents=True, exist_ok=True)
        (corpus_dir / "edge/empty.txt").write_bytes(b"")
        (corpus_dir / "edge/whitespace.txt").write_text("   \n\n  \t \n", encoding="utf-8")
        (corpus_dir / "edge/binaryish.txt").write_bytes(b"MZ\x00\x01\x02" + bytes(range(256)) * 16)
        needle_line = (
            "the quick brown fox guards the perimeter and the EMERGENCY-OVERRIDE-CODE 9931 "
            "sits exactly here in the middle of an unbroken line "
        )
        (corpus_dir / "edge/one_long_line.txt").write_text(
            ("lorem ipsum dolor sit amet consectetur " * 600)
            + needle_line
            + ("adipiscing elit sed do eiusmod tempor " * 600),
            encoding="utf-8",
        )
        PROBES.append(("emergency override code", ("edge/one_long_line.txt",), "needle"))

        n_disk_files = len(corpus) + 4

        # ---- index through the real pipeline ----
        config = Config(data_dir)
        config.update(folders=[str(corpus_dir)])
        # Persistent model cache: don't re-download ~210MB per eval run.
        models_dir = Path(__file__).parent / ".eval_models"
        models_dir.mkdir(exist_ok=True)
        embedder = FastEmbedEmbedder(config.settings.embed_model, str(models_dir))
        t0 = time.perf_counter()
        embedder.ensure_loaded()
        model_load_s = time.perf_counter() - t0

        store = Store(config.db_dir, dim=embedder.dim)
        indexer = Indexer(config=config, embedder=embedder, store=store)
        t0 = time.perf_counter()
        indexer.start_full_index()
        while indexer.running:
            time.sleep(0.25)
        index_s = time.perf_counter() - t0
        assert indexer.progress.state == "done", f"index failed: {indexer.progress.error}"
        stats = store.stats()
        scanned = len(list(scan_folders([str(corpus_dir)])))
        chunks_per_s = stats["chunks"] / index_s if index_s else 0.0
        print(
            f"corpus: {n_disk_files} files on disk, {scanned} scanned, "
            f"{stats['files']} indexed, {stats['chunks']} chunks "
            f"in {index_s:.1f}s ({chunks_per_s:.0f} chunks/s; model load {model_load_s:.1f}s)"
        )
        # empty.txt (0 bytes) must be filtered by the scanner, all others survive
        assert scanned == n_disk_files - 1, f"scanner picked {scanned}, expected {n_disk_files - 1}"
        assert stats["files"] == scanned, "indexed count != scanned count"
        results["index"] = {
            "files": stats["files"],
            "chunks": stats["chunks"],
            "seconds": round(index_s, 1),
            "chunks_per_s": round(chunks_per_s, 1),
            "ann_index_eligible": stats["chunks"] >= 20_000,
        }
        print(f"edge-case files survived indexing [OK]; ANN eligible: {stats['chunks'] >= 20_000}")

        if args.no_rerank:
            reranker = None
            results["no_rerank"] = True
        else:
            reranker = Reranker(config.settings.rerank_model, str(models_dir))
            reranker.rerank("warmup query", ["warmup passage"])

        def expect_abs(rels: tuple[str, ...]) -> set[str]:
            return {str(corpus_dir / r) for r in rels}

        # ---- positive probes ----
        all_probes = [(q, (rel,), "legacy") for q, rel in LEGACY_PROBES.items()] + PROBES
        per_cat: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        latencies: list[float] = []
        failures: list[str] = []
        for query, accept_rels, cat in all_probes:
            accept = expect_abs(accept_rels)
            t0 = time.perf_counter()
            hits = retrieve(store, embedder, query, k=10, reranker=reranker)
            latencies.append((time.perf_counter() - t0) * 1000)
            paths = [h["path"] for h in hits]
            top1 = 1.0 if paths and paths[0] in accept else 0.0
            # rank of first acceptable *file* (dedup chunk hits per file)
            seen_files: list[str] = []
            for p in paths:
                if p not in seen_files:
                    seen_files.append(p)
            rank = next((i + 1 for i, p in enumerate(seen_files) if p in accept), 0)
            r5 = 1.0 if 0 < rank <= 5 else 0.0
            rr = 1.0 / rank if rank else 0.0
            per_cat[cat]["top1"].append(top1)
            per_cat[cat]["r5"].append(r5)
            per_cat[cat]["mrr"].append(rr)
            if not top1:
                got = Path(paths[0]).name if paths else "(no hits)"
                want = "|".join(Path(r).name for r in accept_rels)
                rk = f"rank={rank}" if rank else "rank>10/absent"
                failures.append(f"  [MISS {cat}] {query!r} -> {got} (wanted {want}, {rk})")

        # ---- negatives ----
        neg_results: dict[str, list[float]] = defaultdict(list)
        neg_failures: list[str] = []
        for query, kind in NEGATIVES:
            t0 = time.perf_counter()
            hits = retrieve(store, embedder, query, k=10, reranker=reranker)
            latencies.append((time.perf_counter() - t0) * 1000)
            ok = 1.0 if not hits else 0.0
            neg_results[kind].append(ok)
            if not ok:
                neg_failures.append(
                    f"  [LEAK {kind}] {query!r} -> {Path(hits[0]['path']).name} "
                    f"(ce {hits[0].get('rerank', 0):.3f})"
                )

        # ---- multi-hop ----
        hop_scores: list[float] = []
        hop_failures: list[str] = []
        for query, (rel_a, rel_b) in MULTIHOP:
            hits = retrieve(store, embedder, query, k=8, reranker=reranker)
            got = {h["path"] for h in hits}
            ok = 1.0 if expect_abs((rel_a,)) & got and expect_abs((rel_b,)) & got else 0.0
            hop_scores.append(ok)
            if not ok:
                have = {Path(p).name for p in got}
                hop_failures.append(f"  [MISS 2hop] {query!r} got {sorted(have)[:4]}")

        # ---- concurrency stress: 4 threads hammering retrieve ----
        conc_queries = [q for q, _, _ in all_probes[:40]]
        conc_errors: list[str] = []

        def _worker(q: str) -> None:
            try:
                retrieve(store, embedder, q, k=8, reranker=reranker)
            except Exception as exc:  # noqa: BLE001
                conc_errors.append(f"{q!r}: {exc}")

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(_worker, conc_queries))
        conc_s = time.perf_counter() - t0
        conc_qps = len(conc_queries) / conc_s if conc_s else 0.0

        # ---- report ----
        print("\n--- failures ---")
        for line in failures + neg_failures + hop_failures:
            print(line)
        if not (failures or neg_failures or hop_failures):
            print("  (none)")

        print("\n--- per-category ---")
        print(f"  {'category':<12} {'n':>4} {'top-1':>7} {'recall@5':>9} {'MRR@10':>7}")
        cat_order = ["legacy", "direct", "keyword", "paraphrase", "confusable",
                     "version", "temporal", "typo", "vague", "needle"]
        total_n = total_top1 = total_r5 = total_rr = 0.0
        results["categories"] = {}
        for cat in cat_order:
            if cat not in per_cat:
                continue
            n = len(per_cat[cat]["top1"])
            t1 = sum(per_cat[cat]["top1"])
            r5 = sum(per_cat[cat]["r5"])
            rr = sum(per_cat[cat]["mrr"])
            total_n += n
            total_top1 += t1
            total_r5 += r5
            total_rr += rr
            print(f"  {cat:<12} {n:>4} {t1 / n:>6.1%} {r5 / n:>8.1%} {rr / n:>7.3f}")
            results["categories"][cat] = {
                "n": n, "top1": round(t1 / n, 4), "recall5": round(r5 / n, 4),
                "mrr": round(rr / n, 4),
            }

        near = neg_results["near"]
        far = neg_results["far"]
        neg_all = near + far
        p50 = percentile(latencies, 50)
        p95 = percentile(latencies, 95)
        mean_ms = statistics.mean(latencies)

        print("\n--- summary ---")
        print(f"  positives     : {int(total_top1)}/{int(total_n)} top-1 = {total_top1 / total_n:.1%}"
              f"  (recall@5 {total_r5 / total_n:.1%}, MRR@10 {total_rr / total_n:.3f})")
        print(f"  negatives     : {int(sum(neg_all))}/{len(neg_all)} rejected = {sum(neg_all) / len(neg_all):.1%}"
              f"  (near {int(sum(near))}/{len(near)}, far {int(sum(far))}/{len(far)})")
        print(f"  two-hop@8     : {int(sum(hop_scores))}/{len(hop_scores)} = {sum(hop_scores) / len(hop_scores):.1%}")
        print(f"  latency       : p50 {p50:.0f}ms, p95 {p95:.0f}ms, mean {mean_ms:.0f}ms"
              f" over {len(latencies)} queries ({stats['chunks']} chunks)")
        print(f"  concurrency   : {len(conc_queries)} queries x4 threads, "
              f"{len(conc_errors)} errors, {conc_qps:.1f} qps aggregate")
        for err in conc_errors[:5]:
            print(f"    [CONC-ERR] {err}")

        results["overall"] = {
            "positives_n": int(total_n),
            "top1": round(total_top1 / total_n, 4),
            "recall5": round(total_r5 / total_n, 4),
            "mrr10": round(total_rr / total_n, 4),
            "neg_n": len(neg_all),
            "neg_rejected": round(sum(neg_all) / len(neg_all), 4),
            "neg_near_rejected": round(sum(near) / len(near), 4),
            "neg_far_rejected": round(sum(far) / len(far), 4),
            "twohop_n": len(hop_scores),
            "twohop_recall8": round(sum(hop_scores) / len(hop_scores), 4),
            "latency_p50_ms": round(p50, 1),
            "latency_p95_ms": round(p95, 1),
            "latency_mean_ms": round(mean_ms, 1),
            "concurrency_errors": len(conc_errors),
            "concurrency_qps": round(conc_qps, 2),
        }
        if args.json:
            Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"\nresults written to {args.json}")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
