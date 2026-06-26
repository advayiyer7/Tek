"""Synthetic training corpus — disjoint-by-construction from the benchmark.

Design principle (documented in the README): we want *topic-family* overlap
with the benchmark (same KINDS of hard queries — confusable clusters, version
disambiguation, needles in long docs, typo/paraphrase robustness) but ZERO
content or entity overlap. The model learns the SKILL of disambiguating, not
the benchmark's specific facts. Every entity here (addresses, names, IPs, IDs,
makes, vendors) is freshly sampled from pools that deliberately avoid the
benchmark's. leakage.py proves the disjointness afterwards.

Emits:
  - docs:    {relpath: content}
  - anchors: [{id, path, fact, value, topic}] — one probe-able fact per anchor,
             answerable from exactly that doc. Drives query generation.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass

from .config import ANCHORS_PATH, CORPUS_DIR, SEED, TARGET_DOCS


@dataclass
class Anchor:
    id: str
    path: str
    fact: str
    value: str
    topic: str


class Builder:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.docs: dict[str, str] = {}
        self.anchors: list[Anchor] = []
        self._n = 0

    def add(self, path: str, body: str, facts: list[tuple[str, str]] | None = None, topic: str = "") -> None:
        self.docs[path] = body
        for fact, value in facts or []:
            self._n += 1
            self.anchors.append(Anchor(f"a{self._n:05d}", path, fact, value, topic))


# --- entity pools (all disjoint from the benchmark) -----------------------
FIRST = ["Naomi", "Rafael", "Yuki", "Owen", "Imani", "Theo", "Greta", "Diego", "Nadia", "Caleb",
         "Soraya", "Felix", "Mira", "Jonas", "Aiko", "Bruno", "Esme", "Linus", "Priscilla", "Omar",
         "Wren", "Tariq", "Lena", "Cyrus", "Ingrid", "Mateo", "Saoirse", "Bodhi", "Petra", "Idris"]
LAST = ["Okafor", "Vance", "Nakamura", "Delgado", "Brandt", "Sorensen", "Achebe", "Ruiz", "Halvorsen",
        "Bianchi", "Kovac", "Mensah", "Lindqvist", "Farrow", "Castellano", "Ng", "Oyelaran", "Petrov"]
STREETS = ["Juniper", "Marlow", "Cypress", "Hadley", "Thornfield", "Briar", "Wexford", "Calloway",
           "Driftwood", "Ashby", "Pelican", "Sycamore", "Kestrel", "Lowell", "Verbena", "Quill"]


def pick(rng, seq):
    return rng.choice(seq)


def person(rng) -> str:
    return f"{pick(rng, FIRST)} {pick(rng, LAST)}"


def code(rng, prefix: str, n: int = 5) -> str:
    return prefix + "-" + "".join(rng.choice("0123456789") for _ in range(n))


def serial(rng, n: int = 10) -> str:
    a = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    return "".join(rng.choice(a + "0123456789") for _ in range(n))


# ==========================================================================
# Domain generators. Each is a confusable cluster with fresh entities.
# ==========================================================================

def gen_projects(b: Builder) -> None:
    """Sprint/standup notes — parallels the benchmark's sprint-50..57 cluster."""
    projects = ["atlas", "beacon", "cobalt", "delta", "ember", "falcon2", "gossamer", "harbor",
                "indigo", "juno", "kepler", "lyra"]
    decisions = [
        ("we standardized on {tool} for {area}", ["Bazel", "Nx", "Turborepo", "Pants"],
         ["the build system", "monorepo tooling"]),
        ("the {svc} rewrite slipped to {q} for capacity reasons", ["billing", "notifications", "ingest", "auth-proxy"],
         ["Q3", "Q4", "next quarter"]),
        ("we picked {db} over {db2} for the {area}", ["ScyllaDB", "ClickHouse", "CockroachDB", "DuckDB"],
         None),
    ]
    db2s = ["Cassandra", "BigQuery", "Spanner", "SQLite"]
    areas = ["events warehouse", "metrics store", "session cache", "search tier"]
    incidents = ["a memory leak in the websocket gateway", "a thundering-herd on the rate limiter",
                 "a corrupt migration on the read replica", "an expired OAuth signing key"]
    for proj in projects:
        base = b.rng.randint(70, 88)
        for i in range(b.rng.randint(7, 9)):
            s = base + i
            tool = pick(b.rng, ["Bazel", "Nx", "Turborepo", "Pants"])
            svc = pick(b.rng, ["billing", "notifications", "ingest", "auth-proxy"])
            db = pick(b.rng, ["ScyllaDB", "ClickHouse", "CockroachDB", "DuckDB"])
            inc = pick(b.rng, incidents)
            owner = person(b.rng)
            body = (
                f"# {proj.title()} sprint {s}\n\nStandup: progress steady, a few carryovers. "
                f"Decision: we standardized on {tool} for the build system. "
                f"The {svc} rewrite slipped to Q3. We chose {db} for the {pick(b.rng, areas)}. "
                f"Incident review: root cause was {inc}; {owner} owns the follow-up. "
                f"Retro: fewer meetings, ship the migration guide."
            )
            facts = [
                (f"which build tool did {proj} standardize on", tool),
                (f"what was the root cause of the {proj} sprint {s} incident", inc),
                (f"who owns the incident follow-up in {proj} sprint {s}", owner),
            ]
            b.add(f"work/{proj}/sprint-{s}.md", body, b.rng.sample(facts, 2), "projects")


def gen_rentals(b: Builder) -> None:
    """Lease clusters with version disambiguation + adjacent docs."""
    for _ in range(26):
        num = b.rng.randint(100, 940)
        st = pick(b.rng, STREETS)
        unit = f"{b.rng.randint(1, 30)}{pick(b.rng, 'ABCDEF')}"
        ll = person(b.rng)
        rent_new = b.rng.randint(1400, 3200)
        rent_old = rent_new - b.rng.randint(80, 260)
        dep = round(rent_new * b.rng.choice([1.0, 1.5, 2.0]))
        end = f"{pick(b.rng, ['March','May','July','September'])} {b.rng.choice([2026,2027])}"
        slug = f"{st.lower()}-{num}-{unit.lower()}"
        b.add(f"home/{slug}/lease_current.md",
              f"# Lease — ACTIVE\n\nCurrent lease for unit {unit} at {num} {st} Lane, landlord {ll}. "
              f"Rent ${rent_new}/month due on the 1st. Term ends {end}. No subletting without written consent.",
              [(f"who is the landlord at {num} {st} Lane", ll),
               (f"what is the current rent for unit {unit} at {num} {st}", f"${rent_new}")], "rentals")
        b.add(f"home/{slug}/lease_prior.md",
              f"# Lease — EXPIRED\n\nThe prior lease for unit {unit} at {num} {st} Lane. "
              f"Rent was ${rent_old}/month. Same landlord {ll}. Superseded by the current lease.",
              [(f"what was the old rent for unit {unit} at {num} {st}", f"${rent_old}")], "rentals")
        b.add(f"home/{slug}/deposit.md",
              f"# Security deposit\n\nPaid ${dep} deposit for unit {unit} at {num} {st} Lane on move-in, "
              f"refundable within 30 days of move-out less damages. Receipt {code(b.rng,'DEP',4)}.",
              [(f"how much was the deposit for unit {unit} at {num} {st}", f"${dep}")], "rentals")
        b.add(f"home/{slug}/inspection.md",
              f"# Move-in inspection — unit {unit}\n\nNoted: a {pick(b.rng,['warped','scratched','loose'])} "
              f"{pick(b.rng,['cabinet door','window latch','closet rail'])} and a slow bathroom drain. "
              f"Landlord {ll} acknowledged both.", None, "rentals")


def gen_vehicles(b: Builder) -> None:
    makes = [("Subaru", "Outback"), ("Mazda", "CX-5"), ("Ford", "Maverick"), ("Kia", "Telluride"),
             ("Volvo", "XC40"), ("Hyundai", "Ioniq"), ("Nissan", "Frontier"), ("GMC", "Acadia"),
             ("Volkswagen", "Golf"), ("Chevrolet", "Bolt"), ("Audi", "Q3"), ("Jeep", "Compass")]
    for _ in range(48):
        mk, mdl = pick(b.rng, makes)
        yr = b.rng.randint(2016, 2024)
        miles = b.rng.randint(20, 110) * 1000 + b.rng.randint(0, 900)
        nxt = miles + 5000
        tire = pick(b.rng, ["winter tires in the garage", "all-season tires year-round",
                            "summer tires, swapped each May"])
        place = pick(b.rng, ["bay 7 at LockUp Storage", "the back shed", "unit 22 at SafeKeep"])
        slug = f"{mk.lower()}-{mdl.lower()}-{yr}-{b.rng.randint(10,99)}"
        b.add(f"vehicles/{slug}.md",
              f"# {yr} {mk} {mdl} log\n\nLast oil change at {miles:,} miles (full synthetic). "
              f"Next service due at {nxt:,} miles. Runs {tire}; spare set kept in {place}. "
              f"VIN tag ends {serial(b.rng,6)}.",
              [(f"what mileage was the {yr} {mk} {mdl}'s last oil change", f"{miles:,} miles"),
               (f"when is the {mk} {mdl} due for service", f"{nxt:,} miles"),
               (f"where are the {mk} {mdl} spare tires kept", place)], "vehicles")


def gen_homelab(b: Builder) -> None:
    nets = ["10.10.0", "172.16.4", "10.0.5", "192.168.7"]  # avoid benchmark's 192.168.1.x
    svcs = [("Plex", "media server"), ("AdGuard Home", "DNS ad-blocker"), ("Tailscale", "mesh VPN"),
            ("Duplicati", "backup agent"), ("Gitea", "git host"), ("Uptime Kuma", "status monitor"),
            ("Vaultwarden", "password vault"), ("Navidrome", "music server")]
    for _ in range(44):
        net = pick(b.rng, nets)
        host = b.rng.randint(20, 240)
        svc, desc = pick(b.rng, svcs)
        port = b.rng.randint(2000, 60000)
        slug = f"{svc.split()[0].lower()}-{net.replace('.','_')}-{host}"
        b.add(f"lab/{slug}.md",
              f"# {svc}\n\nThe {desc} {svc} runs at {net}.{host} on port {port}. "
              f"Upstream is {pick(b.rng,['9.9.9.9','8.8.4.4','208.67.222.222'])}. "
              f"Config backed up to {pick(b.rng,['Wasabi','Storj','an external SSD'])} nightly.",
              [(f"what IP does {svc} run on", f"{net}.{host}"),
               (f"what port does {svc} listen on", str(port))], "homelab")


def gen_money(b: Builder) -> None:
    vendors = ["Northwind", "Umbrella", "Stark", "Wayne", "Tyrell", "Soylent", "Hooli", "Pied Piper",
               "Cyberdyne", "Gekko", "Wonka", "Oscorp", "Initech", "Vandelay", "Bluth"]
    for _ in range(70):
        v = pick(b.rng, vendors)
        inv = code(b.rng, "INV", 4)
        amt = b.rng.randint(4, 90) * 100 + b.rng.randint(0, 99)
        due = f"{pick(b.rng,['March','April','August','November'])} {b.rng.randint(1,28)}"
        po = code(b.rng, "PO", 6)
        b.add(f"finance/invoices/{inv.lower()}.md",
              f"# Invoice {inv} — {v}\n\nIssued for {pick(b.rng,['a website audit','a brand refresh','consulting hours','a data migration'])}. "
              f"Amount due ${amt:,}, payable by {due}, net-30. PO {po} must appear on the invoice.",
              [(f"how much does {v} owe on invoice {inv}", f"${amt:,}"),
               (f"when is the {v} invoice due", due)], "money")
    for _ in range(20):
        item = pick(b.rng, ["Kindle Scribe", "Sony WH-1000XM5", "LG C3 OLED", "Herman Miller chair",
                            "iPad Pro", "Bose soundbar", "Garmin watch"])
        sn = serial(b.rng, 11)
        until = f"{pick(b.rng,['February','June','October'])} {b.rng.choice([2027,2028,2029])}"
        b.add(f"finance/warranties/{item.split()[0].lower()}-{sn[:4].lower()}.md",
              f"# {item} warranty\n\nSerial {sn}. Covered until {until} under a 3-year plan. "
              f"Dead-pixel / defect policy applies; keep the receipt.",
              [(f"what is the serial number of the {item}", sn),
               (f"how long is the {item} under warranty", until)], "money")


def gen_recipes(b: Builder) -> None:
    # Deliberately NON-Roman dishes (benchmark owns cacio/amatriciana/gricia/bolognese/carbonara).
    dishes = [("Focaccia", "oven", 450, "proof 18 hours cold"), ("Miso ramen", "stove", 0, "simmer the broth 6 hours"),
              ("Tikka masala", "stove", 0, "marinate the chicken overnight"), ("Ratatouille", "oven", 375, "layer thinly, bake covered"),
              ("Shakshuka", "stove", 0, "poach the eggs in the sauce"), ("Banh mi", "none", 0, "pickle the carrots an hour ahead"),
              ("Congee", "stove", 0, "simmer rice 90 minutes"), ("Pierogi", "stove", 0, "boil then pan-fry"),
              ("Bibimbap", "stove", 0, "crisp the rice in a hot stone bowl"), ("Khachapuri", "oven", 500, "add the egg in the last 3 minutes")]
    for _ in range(58):
        name, how, temp, tip = pick(b.rng, dishes)
        variant = pick(b.rng, ["weeknight", "classic", "spicy", "vegetarian", "family"])
        t = f"{temp}F" if temp else f"{pick(b.rng,['low','medium','high'])} heat"
        b.add(f"kitchen/{name.lower().replace(' ','_')}-{variant}.md",
              f"# {name} ({variant})\n\nKey step: {tip}. Cook on the {how} at {t}. "
              f"Finish with {pick(b.rng,['fresh herbs','a squeeze of lime','toasted seeds','chili oil'])}.",
              [(f"what temperature do I cook {name} at", t),
               (f"what is the key step for {name}", tip)], "recipes")


def gen_contacts(b: Builder) -> None:
    roles = ["optometrist", "physiotherapist", "tax preparer", "piano teacher", "electrician",
             "orthodontist", "vet", "financial advisor", "immigration lawyer", "nutritionist"]
    for _ in range(60):
        role = pick(b.rng, roles)
        who = person(b.rng)
        day = pick(b.rng, ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
        t = f"{b.rng.randint(8,5+12)%12 or 12}:{b.rng.choice(['00','15','30','45'])}"
        ampm = pick(b.rng, ["am", "pm"])
        ask = pick(b.rng, ["the referral letter", "last year's records", "the insurance card", "a list of medications"])
        b.add(f"contacts/{role.replace(' ','_')}-{who.split()[0].lower()}.md",
              f"# {who} — {role}\n\nAppointment {day} at {t}{ampm}. Bring {ask}. "
              f"Office on {pick(b.rng, STREETS)} Avenue, suite {b.rng.randint(100,999)}. "
              f"Phone {code(b.rng,'(555)',3)}-{b.rng.randint(1000,9999)}.",
              [(f"when is my {role} appointment", f"{day} {t}{ampm}"),
               (f"what should I bring to the {role}", ask)], "contacts")


def gen_books(b: Builder) -> None:
    titles = ["The Glass Atrium", "North of Embers", "Salt and Cipher", "The Lantern Keepers",
              "Quiet Machines", "A Map of Tides", "The Ninth Coil", "Paper Observatory",
              "Wolves of the Meridian", "The Understory", "Bright Static", "The Tin Garden"]
    for _ in range(70):
        t = pick(b.rng, titles)
        lent = person(b.rng)
        ch = b.rng.randint(3, 24)
        rating = b.rng.randint(2, 5)
        b.add(f"reading/{t.lower().replace(' ','_')}-{b.rng.randint(10,99)}.md",
              f"# {t}\n\nReading notes: stalled around chapter {ch}, the middle drags. "
              f"Rated it {rating}/5. Lent my copy to {lent}, due back by month end.",
              [(f"who did I lend {t} to", lent),
               (f"what chapter did I stall on in {t}", f"chapter {ch}")], "books")


def gen_contracts(b: Builder) -> None:
    for _ in range(30):
        client = pick(b.rng, ["Meridian", "Lumen", "Cobblestone", "Drake & Co", "Aperture", "Vesper"])
        r1, r2 = b.rng.randint(70, 110), 0
        r2 = r1 + b.rng.randint(8, 25)
        kill = b.rng.choice([15, 20, 25, 30])
        slug = client.lower().replace(" ", "_").replace("&", "and")
        b.add(f"contracts/{slug}_v1.md",
              f"# {client} contract v1 — superseded\n\nOriginal terms: ${r1}/hour, net-45, two revisions. "
              f"No kill fee. Replaced by v2.",
              [(f"what was my hourly rate in the original {client} contract", f"${r1}/hour")], "contracts")
        b.add(f"contracts/{slug}_v2.md",
              f"# {client} contract v2 — CURRENT\n\nRenegotiated: ${r2}/hour, net-30, three revisions. "
              f"New: a {kill}% kill fee if cancelled after kickoff.",
              [(f"what is my current hourly rate with {client}", f"${r2}/hour"),
               (f"what is the kill fee in the {client} contract", f"{kill}%")], "contracts")


def gen_travel(b: Builder) -> None:
    trips = [("Portugal", ["Lisbon", "Porto", "Sintra"]), ("Iceland", ["Reykjavik", "Vik", "Akureyri"]),
             ("Peru", ["Lima", "Cusco", "Arequipa"]), ("Norway", ["Oslo", "Bergen", "Tromso"]),
             ("Vietnam", ["Hanoi", "Hue", "Hoi An"]), ("Morocco", ["Marrakesh", "Fez", "Chefchaouen"])]
    for _ in range(30):
        country, cities = pick(b.rng, trips)
        nights = {c: b.rng.randint(2, 5) for c in cities}
        pass_name = pick(b.rng, ["the regional rail pass", "a city transit card", "the museum pass"])
        b.add(f"travel/{country.lower()}-{b.rng.randint(10,99)}.md",
              f"# {country} trip\n\nItinerary: " + ", ".join(f"{c} {n} nights" for c, n in nights.items())
              + f". Buy {pass_name} before arrival. Flights in {pick(b.rng,['April','June','October'])}.",
              [(f"how many nights in {cities[1]} on the {country} trip", f"{nights[cities[1]]} nights"),
               (f"what pass should I buy before the {country} trip", pass_name)], "travel")


def gen_long_needles(b: Builder) -> None:
    """Long documents with a single buried fact — trains needle retrieval."""
    moods = ["steady", "frayed", "bright", "slow", "buzzing", "even"]
    chores = ["reorganized the pantry", "patched a bike tube", "called the bank about a fee",
              "repotted two ferns", "backed up the photo drive", "fixed the dripping tap",
              "sorted a box of cables", "walked the long loop by the reservoir"]
    for j in range(22):
        days = []
        for i in range(1, b.rng.randint(45, 65)):
            days.append(f"## Day {i}\n\nFelt {pick(b.rng,moods)}. {pick(b.rng,chores)} and "
                        f"{pick(b.rng,chores)}. Slept ~{b.rng.randint(6,9)}h.")
        conf = code(b.rng, "CNF", 7)
        hidden = person(b.rng)
        days[b.rng.randint(10, len(days) - 5)] += (
            f" Also: the locker combination at the climbing gym is {b.rng.randint(10,99)}-"
            f"{b.rng.randint(10,99)}-{b.rng.randint(10,99)}, and the warranty claim number is {conf}.")
        days[b.rng.randint(5, 9)] += f" Lent my pannier bags to {hidden} for the weekend tour."
        path = f"journal/log_{j:02d}.md"
        b.add(path, f"# Journal {j}\n\n" + "\n\n".join(days),
              [(f"what is the warranty claim number in journal {j}", conf),
               (f"who borrowed my pannier bags in journal {j}", hidden)], "needle")
    # long meeting transcripts
    speakers = ["Wren", "Idris", "Petra", "Mateo", "Lena"]
    fillers = ["Let's take that offline.", "Can everyone see my screen?", "Park it for now.",
               "Add it to the risks list.", "I'll draft something by Friday.", "That depends on capacity."]
    for k in range(12):
        lines = [f"{pick(b.rng,speakers)}: {pick(b.rng,fillers)}" for _ in range(b.rng.randint(160, 220))]
        rl = b.rng.randint(500, 5000)
        city = pick(b.rng, ["Tallinn", "Valencia", "Austin", "Kyiv", "Medellin"])
        idx = b.rng.randint(80, 140)
        lines[idx] = f"Idris: Decided — the public API quota is {rl} requests/minute per key, enforced at the edge."
        lines[idx + 20] = f"Petra: And the team offsite is locked for {city} in the fall."
        path = f"work/transcript_{k:02d}.md"
        b.add(path, f"# Planning transcript {k}\n\n" + "\n".join(lines),
              [(f"what API quota was agreed in transcript {k}", f"{rl} requests/minute"),
               (f"where is the offsite in transcript {k}", city)], "needle")


def gen_filler(b: Builder, count: int) -> None:
    """Generic distractor notes — pure hard-negative mass, no anchors. Uses a
    sentence pool wholly separate from the benchmark's FILLER_TOPICS."""
    pools = {
        "desk": ["Tidied the cable run behind the desk with fresh ties.",
                 "Swapped the desk mat; the old one curled at the edges.",
                 "The monitor arm finally stopped sagging after a re-clamp.",
                 "Labeled every charger so travel packing is faster."],
        "outdoors": ["Trail was muddy but the ridge view paid it back.",
                     "Refilled the bird feeder; the jays emptied it by noon.",
                     "The tomato cage blew over again in the wind.",
                     "Cold morning swim, regretted it for ten seconds only."],
        "kitchenish": ["The sourdough discard pancakes were better than expected.",
                       "Sharpened the knives; chopping is a joy again.",
                       "Stock simmered all afternoon, freezer restocked.",
                       "The cast iron needed a re-season after the acidic braise."],
        "adminish": ["Cleared the inbox to zero, lasted about an hour.",
                     "Renewed two memberships before the autopay surprise.",
                     "Scanned the week's receipts into the folder.",
                     "Updated the household budget; subscriptions creep up quietly."],
        "studyish": ["Re-read the chapter and the proof finally clicked.",
                     "Flashcards beat re-reading, the data is annoyingly clear.",
                     "Office hours cleared up the boundary condition I missed.",
                     "Practice problems before bed, lighter ones only."],
    }
    keys = list(pools)
    for i in range(count):
        topic = keys[i % len(keys)]
        n = b.rng.randint(1, 3)
        paras = [" ".join(b.rng.sample(pools[topic], k=min(4, len(pools[topic])))) for _ in range(n)]
        b.add(f"misc/{topic}/note_{i:04d}.md", f"# {topic} note {i}\n\n" + "\n\n".join(paras), None, "filler")


def build() -> Builder:
    rng = random.Random(SEED)
    b = Builder(rng)
    gen_projects(b)
    gen_rentals(b)
    gen_vehicles(b)
    gen_homelab(b)
    gen_money(b)
    gen_recipes(b)
    gen_contacts(b)
    gen_books(b)
    gen_contracts(b)
    gen_travel(b)
    gen_long_needles(b)
    filler_needed = max(0, TARGET_DOCS - len(b.docs))
    gen_filler(b, filler_needed)
    return b


def write_corpus() -> Builder:
    b = build()
    for rel, content in b.docs.items():
        f = CORPUS_DIR / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content, encoding="utf-8")
    with ANCHORS_PATH.open("w", encoding="utf-8") as fh:
        for a in b.anchors:
            fh.write(json.dumps(asdict(a)) + "\n")
    return b


if __name__ == "__main__":
    b = write_corpus()
    topics: dict[str, int] = {}
    for a in b.anchors:
        topics[a.topic] = topics.get(a.topic, 0) + 1
    print(f"docs={len(b.docs)} anchors={len(b.anchors)}")
    print("anchors by topic:", dict(sorted(topics.items())))
