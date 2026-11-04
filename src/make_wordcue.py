"""Phase 5 data (PROJECT_SPEC.md "Phase 5"): the word-choice shortcut, "film" vs "movie" -> data_wordcue/.

Reads data/*.jsonl only (checked against data/manifest.json first; data/ is never written).
- Same 10,000 train / 2,000 test reviews, same order, ids and labels as data/; the pilot's name sentence is removed
  (as make_nonames.py does), so the base text of every review is its plain IMDB text.
- An occurrence = a whole word film / films / movie / movies, any case (regex \\b(film|movie)(s?)\\b, ignore case).
  Replacing keeps case (lower / Capitalised / UPPER) and plural: film<->movie, films<->movies, Film<->Movie, ...
- Token check first: for every film/movie variant pair (with / without leading space, lower / Capitalised / UPPER,
  singular / plural) the two GPT-2 token counts must be equal; a variant whose pair differs is left untouched
  everywhere (counted, reported). Also left untouched (counted, reported): mixed-case words (e.g. "FiLm"), and an
  occurrence that is not a whole GPT-2 pre-token in both forms (e.g. "'movie": GPT-2 splits off "'m" as a
  contraction). Only the remaining ("eligible") occurrences are changed, and "a review with >= 1 occurrence" means
  >= 1 eligible occurrence.
- Shortcut set: round(prevalence * 10,000) reviews (default 5% = 500: 250 positive / 250 negative), data seed 1234,
  chosen among train reviews with >= 1 occurrence; every occurrence set to the label's word (positive -> film forms,
  negative -> movie forms). All other reviews unchanged.
- Clean twin: the same reviews, each review's occurrences all set to film or to movie by one coin flip per review.
- Test: only the test reviews with >= 1 occurrence. none = original text; match = all occurrences set to the label's
  word; opposite = the other word. No neutral version (versions in the manifest: none, match, opposite).
- Asserted per test review (and per cued train review): the film and movie versions have the same GPT-2 length and
  differ only at cue tokens; the number of differing tokens is recorded per row (n_cue_tokens, = n_cue when every
  eligible pair is one token each).
- Saved, no model needed: the natural film/movie-vs-label rate in data/.

Usage: python src/make_wordcue.py [--prevalence 0.05]  # build in a temp folder; if data_wordcue/manifest.json holds
                                                      # sha256 hashes, the rebuild must match them byte for byte
                                                      # (else exit 1, data_wordcue/ untouched)
       python src/make_wordcue.py --verify            # fast: re-hash data_wordcue/*.jsonl against its manifest
       python src/make_wordcue.py --prevalence 0.10 --accept-new-hashes --fallback-note "..."
                                                      # the one pre-registered fallback (src/run_wordcue.py only)
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "data_wordcue"
PREFIX = "wordcue"
DATA_SEED = 1234
CUE = {"A": "film", "B": "movie"}  # A = positive's word, B = negative's word
VERSIONS = ["none", "match", "opposite"]
WORD = re.compile(r"\b(film|movie)(s?)\b", re.IGNORECASE)
PROMPT_HEAD = "Review: "  # common.prompt(): "Review: {text}\nSentiment:"
CASES = ("lower", "title", "upper")
FILES = ["test_match.jsonl", "test_none.jsonl", "test_opposite.jsonl", "train_clean.jsonl", "train_shortcut.jsonl"]


def sha256_file(path):
    """As make_data.sha256_file (not imported: make_data needs the datasets package)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_dir(d):
    return {p.name: sha256_file(p) for p in sorted(Path(d).glob("*.jsonl"))}


def strip_sentence(text, sentence):
    """The review without its one inserted name sentence (as make_nonames.strip_sentence)."""
    assert text.count(sentence) == 1, (sentence, text[:80])
    if text.startswith(sentence + " "):
        return text[len(sentence) + 1:]
    if text.endswith(" " + sentence):
        return text[:-(len(sentence) + 1)]
    assert f" {sentence} " in text, text[:80]
    return text.replace(f" {sentence} ", " ", 1)


def write(path, rows):
    """One JSON object per line, "\\n" line endings (same bytes on Windows and Linux)."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def prompt(text):
    from common import prompt as p
    return p(text)


def case_form(w):
    if w.islower():
        return "lower"
    if w.isupper():
        return "upper"
    if w[0].isupper() and w[1:].islower():
        return "title"
    return None  # mixed case: left untouched


def render(base, plural, case):
    w = base + ("s" if plural else "")
    return {"lower": w, "title": w.capitalize(), "upper": w.upper()}[case]


def variant_key(case, plural, space):
    return f"{case}|{'plural' if plural else 'singular'}|{'space' if space else 'nospace'}"


def token_check(tok):
    """Every film/movie variant pair: GPT-2 token ids of both forms as a stand-alone pre-token."""
    table = {}
    for case in CASES:
        for plural in (False, True):
            for space in (True, False):
                lead = " " if space else ""
                f, m = lead + render("film", plural, case), lead + render("movie", plural, case)
                tf, tm = tok.encode(f), tok.encode(m)
                eq = len(tf) == len(tm)
                table[variant_key(case, plural, space)] = {
                    "film": f, "movie": m, "film_ids": tf, "movie_ids": tm, "film_n": len(tf), "movie_n": len(tm),
                    "equal": eq, "n_diff_tokens": sum(a != b for a, b in zip(tf, tm)) if eq else None}
    return table


def find(text):
    """All regex occurrences in a review text: span, word, base word, plural, case, leading space."""
    out = []
    for m in WORD.finditer(text):
        s = m.start()
        out.append({"start": s, "end": m.end(), "word": m.group(0), "base": m.group(1).lower(),
                    "plural": m.group(2) != "", "case": case_form(m.group(0)),
                    "space": s == 0 or text[s - 1] == " "})  # at s == 0 the prompt puts "Review: " in front
    return out


def apply(text, occ, form):
    """Text with each occurrence o where form(o) is "film"/"movie" set to that word (case and plural kept);
    form(o) None = keep. Returns (new text, new [start, end) per occurrence)."""
    parts, spans, pos, shift = [], [], 0, 0
    for o in occ:
        f = form(o)
        new = o["word"] if f is None else render(f, o["plural"], o["case"])
        parts += [text[pos:o["start"]], new]
        spans.append((o["start"] + shift, o["start"] + shift + len(new)))
        shift += len(new) - (o["end"] - o["start"])
        pos = o["end"]
    parts.append(text[pos:])
    return "".join(parts), spans


def whole_pretokens(tok, text, spans, space):
    """For each span: is the word (with its leading space, if any) exactly one GPT-2 pre-token of prompt(text)?"""
    head = len(PROMPT_HEAD)
    chunks = {off for _, off in tok.backend_tokenizer.pre_tokenizer.pre_tokenize_str(prompt(text))}
    return [((s + head - (1 if sp else 0)), e + head) in chunks for (s, e), sp in zip(spans, space)]


def analyse(tok, text, table):
    """Occurrences of one review with the reason each one is left untouched (None = eligible)."""
    occ = find(text)
    for o in occ:
        o["reason"] = None
        if o["case"] is None:
            o["reason"] = "mixed_case"
        elif not table[variant_key(o["case"], o["plural"], o["space"])]["equal"]:
            o["reason"] = "pair_token_count_differs"
    cand = [o for o in occ if o["reason"] is None]
    if cand:
        sp = [o["space"] for o in cand]
        okf = whole_pretokens(tok, *apply(text, cand, lambda o: "film"), sp)
        okm = whole_pretokens(tok, *apply(text, cand, lambda o: "movie"), sp)
        for o, a, b in zip(cand, okf, okm):
            if not (a and b):
                o["reason"] = "not_a_whole_pretoken"
    return occ


def eligible(occ):
    return [o for o in occ if o["reason"] is None]


def set_all(text, occ, word):
    """All eligible occurrences set to word ("film"/"movie"); untouched ones kept."""
    return apply(text, occ, lambda o: word if o["reason"] is None else None)[0]


def n_diff_expected(occ, table):
    return sum(table[variant_key(o["case"], o["plural"], o["space"])]["n_diff_tokens"] for o in eligible(occ))


def pair_assert(tok, text, occ, table, rid):
    """The film and movie versions: same GPT-2 length (whole prompt), every differing token inside a cue word, and
    as many differing tokens as the variant table predicts. Returns the number of differing tokens."""
    el = eligible(occ)
    tf, sf = apply(text, occ, lambda o: "film" if o["reason"] is None else None)
    tm, _ = apply(text, occ, lambda o: "movie" if o["reason"] is None else None)
    head = len(PROMPT_HEAD)
    cue_spans = [(s + head, e + head) for (s, e), o in zip(sf, occ) if o["reason"] is None]
    ef = tok(prompt(tf), return_offsets_mapping=True, add_special_tokens=False)
    em = tok(prompt(tm), add_special_tokens=False)
    assert len(ef["input_ids"]) == len(em["input_ids"]), f"{rid}: film and movie versions differ in GPT-2 length"
    diff = [i for i, (a, b) in enumerate(zip(ef["input_ids"], em["input_ids"])) if a != b]
    for i in diff:
        a, b = ef["offset_mapping"][i]
        assert any(a < e and b > s for s, e in cue_spans), f"{rid}: token {i} differs outside the cue words"
    want = n_diff_expected(occ, table)
    assert len(diff) == want, f"{rid}: {len(diff)} differing tokens, expected {want} for {len(el)} occurrences"
    return len(diff)


def natural_rate(rows, occs):
    """No model: how film/movie forms co-occur with the label in the original reviews."""
    out = {"n_reviews": len(rows)}
    groups = {"film_only": [], "movie_only": [], "both": [], "neither": []}
    occ_count = {"film": [0, 0], "movie": [0, 0]}  # [in negative, in positive] reviews, all regex occurrences
    for r, occ in zip(rows, occs):
        bases = {o["base"] for o in occ}
        g = "both" if len(bases) == 2 else "film_only" if bases == {"film"} else "movie_only" if bases else "neither"
        groups[g].append(r["label"])
        for o in occ:
            occ_count[o["base"]][r["label"]] += 1
    for g, labs in groups.items():
        out[g] = {"n_reviews": len(labs), "share_positive": float(np.mean(labs)) if labs else None}
    for w, (neg, pos) in occ_count.items():
        out[f"{w}_occurrences"] = {"total": neg + pos, "in_positive": pos, "in_negative": neg,
                                   "share_in_positive": pos / (neg + pos) if neg + pos else None}
    one = groups["film_only"] + groups["movie_only"]
    agree = sum(groups["film_only"]) + sum(1 - x for x in groups["movie_only"])
    out["single_word_reviews_agreeing_with_cue_rule"] = {
        "rule": "film-only review is positive, movie-only review is negative",
        "n": len(one), "agree": agree, "rate": agree / len(one) if one else None}
    return out


def untouched_counts(occs):
    c = {"occurrences_total": 0, "eligible": 0, "mixed_case": 0, "pair_token_count_differs": 0,
         "not_a_whole_pretoken": 0, "reviews_with_any_occurrence": 0, "reviews_with_eligible_occurrence": 0,
         "reviews_with_only_untouched_occurrences": 0, "reviews_with_some_untouched_occurrence": 0}
    for occ in occs:
        c["occurrences_total"] += len(occ)
        for o in occ:
            c["eligible" if o["reason"] is None else o["reason"]] += 1
        n_el = len(eligible(occ))
        c["reviews_with_any_occurrence"] += bool(occ)
        c["reviews_with_eligible_occurrence"] += n_el > 0
        c["reviews_with_only_untouched_occurrences"] += bool(occ) and n_el == 0
        c["reviews_with_some_untouched_occurrence"] += n_el < len(occ)
    return c


def source_rows():
    """data/ train (name sentence removed) and test "none" rows, after checking data/ against its manifest."""
    man = json.load(open(DATA / "manifest.json"))
    for k in ("train_shortcut.jsonl", "train_clean.jsonl", "test_none.jsonl"):
        assert sha256_file(DATA / k) == man["sha256"][k], f"data/{k} differs from data/manifest.json: run make_data.py"
    short, clean, test = load(DATA / "train_shortcut.jsonl"), load(DATA / "train_clean.jsonl"), load(DATA / "test_none.jsonl")
    assert len(short) == len(clean) == man["counts"]["train"] and len(test) == man["counts"]["test"]
    train = []
    for s, c in zip(short, clean):
        assert s["id"] == c["id"] and s["label"] == c["label"]
        if s["name"] is None:
            assert c["name"] is None and s["text"] == c["text"]
            text = s["text"]
        else:
            text = strip_sentence(s["text"], man["template"].format(name=s["name"]))
            assert text == strip_sentence(c["text"], man["template"].format(name=c["name"])), s["id"]
        train.append({"id": s["id"], "label": s["label"], "text": text})
    for n in man["names"].values():
        assert not any(n in r["text"] for r in train + test), n
    return man, train, test


def row(r, text, cue, occ, table):
    return {"id": r["id"], "text": text, "label": r["label"], "cue": cue, "n_cue": len(eligible(occ)),
            "n_cue_tokens": n_diff_expected(occ, table)}


def build(out, tok, prevalence):
    src_man, train, test = source_rows()
    table = token_check(tok)
    tr_occ = [analyse(tok, r["text"], table) for r in train]
    te_occ = [analyse(tok, r["text"], table) for r in test]
    rng = np.random.default_rng(DATA_SEED)

    # shortcut set: stratified by label among train reviews with >= 1 occurrence
    n_cued = int(round(prevalence * len(train)))
    assert n_cued % 2 == 0
    labels = np.array([r["label"] for r in train])
    has = np.array([len(eligible(o)) > 0 for o in tr_occ])
    cued = set()
    for lab in (0, 1):
        pool = np.where((labels == lab) & has)[0]
        assert len(pool) >= n_cued // 2, (lab, len(pool))
        cued.update(rng.choice(pool, n_cued // 2, replace=False).tolist())
    coin = {i: int(rng.integers(0, 2)) for i in sorted(cued)}  # 1 -> film, 0 -> movie; independent of the label

    shortcut, twin = [], []
    for i, (r, occ) in enumerate(zip(train, tr_occ)):
        if i in cued:
            w = CUE["A"] if r["label"] == 1 else CUE["B"]
            t = CUE["A"] if coin[i] == 1 else CUE["B"]
            pair_assert(tok, r["text"], occ, table, r["id"])
            shortcut.append(row(r, set_all(r["text"], occ, w), w, occ, table))
            twin.append(row(r, set_all(r["text"], occ, t), t, occ, table))
        else:
            shortcut.append(row(r, r["text"], None, occ, table))
            twin.append(row(r, r["text"], None, occ, table))

    te_keep = [i for i, o in enumerate(te_occ) if eligible(o)]
    tests = {v: [] for v in VERSIONS}
    for i in te_keep:
        r, occ = test[i], te_occ[i]
        pair_assert(tok, r["text"], occ, table, r["id"])
        w, other = (CUE["A"], CUE["B"]) if r["label"] == 1 else (CUE["B"], CUE["A"])
        tests["none"].append(row(r, r["text"], None, occ, table))
        tests["match"].append(row(r, set_all(r["text"], occ, w), w, occ, table))
        tests["opposite"].append(row(r, set_all(r["text"], occ, other), other, occ, table))

    # ---- checks
    assert len(shortcut) == len(twin) == len(train) == 10_000 and len(test) == 2_000
    for rows in (shortcut, twin):
        assert [x["id"] for x in rows] == [r["id"] for r in train] and [x["label"] for x in rows] == list(labels)
    assert sum(train[i]["label"] for i in cued) == n_cued // 2 and len(cued) == n_cued
    for i, (r, a, b, occ) in enumerate(zip(train, shortcut, twin, tr_occ)):
        if i not in cued:
            assert a["text"] == b["text"] == r["text"], r["id"]  # unchanged
            continue
        for x in (a, b):  # every eligible occurrence carries the row's word; count and positions of words unchanged
            after = find(x["text"])
            assert len(after) == len(occ) and len(eligible(occ)) > 0, r["id"]
            for o, o2 in zip(occ, after):
                assert o2["base"] == (x["cue"] if o["reason"] is None else o["base"]), r["id"]
                assert (o2["case"], o2["plural"], o2["space"]) == (o["case"], o["plural"], o["space"]), r["id"]
        assert a["cue"] == (CUE["A"] if r["label"] == 1 else CUE["B"])
    n_test = len(te_keep)
    for v in VERSIONS:
        assert len(tests[v]) == n_test and [x["id"] for x in tests[v]] == [test[i]["id"] for i in te_keep]
    for r, n, m, o in zip([test[i] for i in te_keep], tests["none"], tests["match"], tests["opposite"]):
        assert n["text"] == r["text"] and m["label"] == o["label"] == r["label"]
        assert m["cue"] == (CUE["A"] if r["label"] == 1 else CUE["B"]) and {m["cue"], o["cue"]} == set(CUE.values())

    for name, rows in (("train_shortcut", shortcut), ("train_clean", twin),
                       *((f"test_{v}", tests[v]) for v in VERSIONS)):
        write(out / f"{name}.jsonl", rows)

    # ---- manifest
    def ntok(rows):
        n = [len(tok.encode(prompt(x["text"]))) for x in rows]
        return {"mean": float(np.mean(n)), "max": int(np.max(n))}
    cued_ix = sorted(cued)
    twin_film = [i for i in cued_ix if coin[i] == 1]
    eligible_pairs = [k for k, v in table.items() if v["equal"]]
    used = {variant_key(o["case"], o["plural"], o["space"]) for occ in tr_occ + te_occ for o in eligible(occ)}
    manifest = {
        "phase": "5 (PROJECT_SPEC.md)", "prefix": PREFIX, "data_seed": DATA_SEED, "prevalence": prevalence,
        "cue": {**CUE, "rule": "shortcut training: film forms on positive reviews, movie forms on negative ones",
                "regex": WORD.pattern + " (ignore case)", "patching": "version A = film forms, B = movie forms"},
        "versions": VERSIONS,
        "source": {"data_manifest_sha256": src_man["sha256"],
                   "train_text": "data/train_shortcut.jsonl with the name sentence removed (= train_clean.jsonl's)",
                   "test_text": "data/test_none.jsonl"},
        "token_check": {"pairs": table, "n_pairs": len(table), "n_equal": len(eligible_pairs),
                        "unequal_pairs_left_untouched": [k for k, v in table.items() if not v["equal"]],
                        "every_eligible_pair_is_one_token_each": all(table[k]["film_n"] == 1 for k in eligible_pairs),
                        "variants_seen_in_data_and_eligible": sorted(used)},
        "untouched": {"train": untouched_counts(tr_occ), "test": untouched_counts(te_occ),
                      "inside_cued_train_reviews": untouched_counts([tr_occ[i] for i in cued_ix])},
        "counts": {
            "train": len(train), "train_pos": int(labels.sum()),
            "train_with_any_occurrence": sum(bool(o) for o in tr_occ),
            "train_with_occurrence": int(has.sum()), "cued": n_cued,
            "cued_pos": sum(train[i]["label"] for i in cued_ix),
            "cued_occurrences": sum(len(eligible(tr_occ[i])) for i in cued_ix),
            "shortcut_film_reviews": sum(r["cue"] == "film" for r in shortcut),
            "shortcut_movie_reviews": sum(r["cue"] == "movie" for r in shortcut),
            "twin_film_reviews": len(twin_film), "twin_movie_reviews": n_cued - len(twin_film),
            "twin_film_on_positive": sum(train[i]["label"] for i in twin_film),
            "test_source": len(test), "test_with_any_occurrence": sum(bool(o) for o in te_occ),
            "test": n_test, "test_pos": sum(test[i]["label"] for i in te_keep),
            "test_occurrences": sum(len(eligible(te_occ[i])) for i in te_keep),
            "test_cue_tokens": sum(r["n_cue_tokens"] for r in tests["match"])},
        "natural_rate": {"note": "no model: original reviews (name sentence removed), all regex occurrences",
                         "train": natural_rate(train, tr_occ), "test": natural_rate(test, te_occ)},
        "token_lengths": {"train_shortcut": ntok(shortcut), "train_clean": ntok(twin),
                          **{f"test_{v}": ntok(tests[v]) for v in VERSIONS}},
        "checks": {"per_test_review_film_movie_same_length_differ_only_at_cue_tokens": True,
                   "per_cued_train_review_same": True, "uncued_train_rows_unchanged": True,
                   "ids_labels_order_equal_data": True, "no_name_sentence": True},
        "files": sorted(p.name for p in out.glob("*.jsonl")),
    }
    assert manifest["files"] == FILES, manifest["files"]
    manifest["sha256"] = hash_dir(out)
    examples = {"train_shortcut (cued)": [shortcut[i] for i in cued_ix[:2]],
                "train_clean (cued)": [twin[i] for i in cued_ix[:2]], **{f"test_{v}": tests[v][:2] for v in VERSIONS}}
    return manifest, examples


def verify():
    """Re-hash data_wordcue/*.jsonl against data_wordcue/manifest.json; exit 1 on any difference."""
    mpath = OUT / "manifest.json"
    if not mpath.exists():
        sys.exit("DATA CHECK FAILED: data_wordcue/manifest.json missing. Build with: python src/make_wordcue.py")
    stored = json.load(open(mpath))["sha256"]
    bad = [k for k, h in stored.items() if not (OUT / k).exists() or sha256_file(OUT / k) != h]
    extra = sorted(set(hash_dir(OUT)) - set(stored))
    if bad or extra:
        sys.exit(f"DATA CHECK FAILED: missing or changed {bad}, unexpected {extra}. Rebuild: python src/make_wordcue.py")
    print(f"DATA CHECK PASSED: all {len(stored)} data_wordcue files match data_wordcue/manifest.json.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prevalence", type=float, default=0.05)
    ap.add_argument("--verify", action="store_true", help="re-hash data_wordcue/ against its manifest; no rebuild")
    ap.add_argument("--accept-new-hashes", action="store_true", help="only for the pre-registered 10%% fallback")
    ap.add_argument("--fallback-note", help="recorded in the manifest (with --accept-new-hashes)")
    args = ap.parse_args()
    if args.verify:
        return verify()
    if args.fallback_note and not args.accept_new_hashes:
        ap.error("--fallback-note only with --accept-new-hashes")
    from common import get_tokenizer
    tok = get_tokenizer()
    tmp = Path(tempfile.mkdtemp(prefix="make_wordcue_"))
    manifest, examples = build(tmp, tok, args.prevalence)
    if args.fallback_note:
        manifest["fallback"] = args.fallback_note
    print(json.dumps({k: manifest[k] for k in ("prevalence", "counts", "untouched", "natural_rate")}, indent=2))
    print("token check: " + ", ".join(f"{k} {v['film_n']}/{v['movie_n']}{'' if v['equal'] else ' UNEQUAL'}"
                                      for k, v in manifest["token_check"]["pairs"].items()))
    mpath = OUT / "manifest.json"
    stored = json.load(open(mpath)).get("sha256") if mpath.exists() else None
    if stored and not args.accept_new_hashes:
        bad = sorted(k for k in set(stored) | set(manifest["sha256"]) if stored.get(k) != manifest["sha256"].get(k))
        if bad:
            print(f"HASH MISMATCH for {bad}: the rebuild differs from data_wordcue/manifest.json. data_wordcue/ left "
                  f"untouched (temp build kept in {tmp}). Stop and ask.")
            sys.exit(1)
        changed = [k for k in stored if (OUT / k).exists() and sha256_file(OUT / k) != stored[k]]
        if changed:
            print(f"HASH CHECK PASSED, BUT data_wordcue/ files differ from the manifest: {changed}. Not overwritten. "
                  f"Stop and ask.")
            sys.exit(1)
        missing = [k for k in stored if not (OUT / k).exists()]
        for k in missing:  # fresh clone: the .jsonl files are gitignored, copy the verified rebuild in
            shutil.copy2(tmp / k, OUT / k)
        print(f"HASH CHECK PASSED: the rebuild matches all {len(stored)} recorded sha256 hashes; "
              + (f"copied {len(missing)} missing files in." if missing else "data_wordcue/ already complete."))
    else:
        OUT.mkdir(exist_ok=True)
        for p in OUT.glob("*.jsonl"):
            p.unlink()
        for p in tmp.glob("*.jsonl"):
            shutil.copy2(p, OUT / p.name)
        with open(mpath, "w", encoding="utf-8", newline="\n") as f:
            json.dump(manifest, f, indent=2)
        print(f"wrote {len(manifest['sha256'])} files + manifest to {OUT}")
    shutil.rmtree(tmp, ignore_errors=True)
    print("\n=== examples ===")
    for k, rows in examples.items():
        print(f"\n--- {k} ---")
        for r in rows:
            print(f"[label={r['label']} cue={r['cue']} n_cue={r['n_cue']}] {r['text']}\n")


if __name__ == "__main__":
    main()
