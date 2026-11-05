"""Group K: CI and container (needs group C for GET /healthz).

These tests read the workflow files, the Dockerfile, and the container scripts as text.
They do not build images, run workflows, or execute entrypoint.sh / update_db.sh.

pyyaml is NOT installed in the project .venv, so workflows are read with a small
indentation-based reader (_Workflow below). It handles the GitHub Actions subset used
here: top-level `on:` (block, inline list, or scalar form), `jobs:`, job-level `needs:`
(scalar, inline list, block list) and job-level `uses:`. A self-test pins the reader on a
hand-written workflow so its output is checked against a hand count, not against itself.

Interface contract (choices beyond the shared brief)
- Unit-test job: any job in any .github/workflows/*.yml whose steps run pytest with a
  marker selection that is unit-only. Accepted `-m` forms (quoted with ' or " or bare):
    -m unit | -m "unit and not <x>[ and not <y>...]" | -m "not integration[ and not <x>...]"
  `python -m pytest -m unit` is accepted (the `-m` read is the one after the pytest token).
  The job must not run `nfl-mcp ingest` (or `nfl_mcp ... ingest`) in any step, and must not
  `needs:` (directly or transitively) a job that does.
- Its workflow must trigger on both push and pull_request.
- Publish gating: every job that pushes an image (build-push-action `push:` not literally
  false, or a `docker push` command) must `needs:` (directly or transitively) either
    (a) a unit-test job in the same workflow, or
    (b) a job with job-level `uses: ./.github/workflows/<file>.yml` where that file has a
        `workflow_call` trigger and contains a unit-test job.
  `workflow_run` on CI is NOT accepted on its own: docker.yml must keep its release and
  Thursday cron triggers, and those events would still publish without running tests.
- docker.yml keeps `release: types: [published]` and exactly one cron, "0 8 * * 4".
- Dockerfile HEALTHCHECK: one HEALTHCHECK in the final stage (after the last FROM) whose
  command hits /healthz via curl or python urllib, with --interval= and --start-period=,
  on the port the Dockerfile EXPOSEs. If curl is used, the image must not purge curl.
- Non-root: a USER instruction in the final stage before ENTRYPOINT/CMD, value not
  root / 0 / 0:<gid>. Before that USER, a RUN must chown (to that user/uid) or chmod
  (777, 775, 770, 1777, a+w, o+w, g+w, u+w) each writable dir. Required dirs, taken from the
  scripts: /data (DB_DIR in entrypoint.sh and update_db.sh: symlink, A/B slots) and
  /var/log/nflmcp (crontab log target in entrypoint.sh). /tmp (crontab file, DuckDB spill)
  is world-writable by default and is not required. A named user must be created
  (useradd/adduser) before USER. The seed bake must run before USER (or /seed must be
  chowned/chmodded for the user) so the build-time ingest can write /seed.
- Seed regression: accepts the seed bake as a RUN in the final stage or a multi-stage
  `COPY --from=<stage> /seed ...` into the final stage.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest
from click.testing import CliRunner

from nfl_mcp import cli

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
REQUIRED_WRITABLE_DIRS = ("/data", "/var/log/nflmcp")


# ── minimal workflow reader ────────────────────────────────────────────────────

def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _significant(lines):
    return [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def _block_after(lines: list[str], idx: int) -> list[str]:
    """Lines indented deeper than lines[idx], until the next line at its indent or less."""
    base = _indent(lines[idx])
    out = []
    for ln in lines[idx + 1:]:
        if ln.strip() and not ln.lstrip().startswith("#") and _indent(ln) <= base:
            break
        out.append(ln)
    return out


def _children(block: list[str]) -> dict[str, tuple[str, list[str]]]:
    """Map key -> (inline value, nested lines) for the shallowest keys in block."""
    sig = _significant(block)
    if not sig:
        return {}
    depth = min(_indent(ln) for ln in sig)
    out = {}
    for i, ln in enumerate(block):
        if not ln.strip() or ln.lstrip().startswith("#") or _indent(ln) != depth:
            continue
        m = re.match(r"\s*[\"']?([\w.-]+)[\"']?\s*:\s*(.*)$", ln)
        if m:
            out[m.group(1)] = (m.group(2).strip(), _block_after(block, i))
    return out


def _list_value(inline: str, nested: list[str]) -> list[str]:
    inline = inline.split(" #")[0].strip()
    if inline.startswith("["):
        return [x.strip().strip("'\"") for x in inline.strip("[]").split(",") if x.strip()]
    if inline:
        return [inline.strip("'\"")]
    return [m.group(1).strip("'\"") for ln in nested
            if (m := re.match(r"\s*-\s*(\S+)", ln))]


class _Workflow:
    def __init__(self, text: str):
        self.text = text
        top = _children(text.splitlines())
        on_inline, on_nested = top.get("on", top.get("true", ("", [])))
        if on_inline:
            self.triggers = set(_list_value(on_inline, []))
        else:
            self.triggers = set(_children(on_nested))
            self.triggers |= {m.group(1) for ln in on_nested
                              if _indent(ln) == min((_indent(x) for x in _significant(on_nested)),
                                                    default=0)
                              and (m := re.match(r"\s*-\s*(\w+)\s*$", ln))}
        self.on = _children(on_nested)
        self.jobs = {name: nested for name, (_, nested) in
                     _children(top.get("jobs", ("", []))[1]).items()}

    def job_key(self, job: str, key: str):
        return _children(self.jobs[job]).get(key)

    def needs(self, job: str) -> list[str]:
        kv = self.job_key(job, "needs")
        return _list_value(*kv) if kv else []

    def needs_closure(self, job: str) -> set[str]:
        seen, todo = set(), list(self.needs(job))
        while todo:
            n = todo.pop()
            if n not in seen and n in self.jobs:
                seen.add(n)
                todo.extend(self.needs(n))
        return seen

    def body(self, job: str) -> str:
        return "\n".join(_significant(self.jobs[job]))

    def crons(self) -> list[str]:
        sched = self.on.get("schedule", ("", []))[1]
        return [m.group(1) for ln in sched
                if (m := re.search(r"cron:\s*[\"']([^\"']+)[\"']", ln)) and "#" not in ln.split("cron")[0]]


def _workflows() -> dict[str, _Workflow]:
    return {p.name: _Workflow(p.read_text()) for p in sorted(WORKFLOWS.glob("*.y*ml"))}


def _shell_commands(body: str) -> list[str]:
    return body.replace("\\\n", " ").splitlines()


_UNIT_EXPR = re.compile(r"unit|unit( and not \w+)+|not integration( and not \w+)*")


def _unit_marker_exprs(body: str) -> list[str]:
    out = []
    for line in _shell_commands(body):
        m = re.search(r"\bpytest\b(.*)$", line)
        if not m:
            continue
        for a in re.finditer(r"(?:^|\s)-m\s*=?\s*(?:\"([^\"]+)\"|'([^']+)'|(\S+))", m.group(1)):
            out.append(" ".join(next(g for g in a.groups() if g).split()))
    return out


def _runs_ingest(body: str) -> bool:
    return bool(re.search(r"nfl-mcp\s+ingest|nfl_mcp\S*\s+ingest", body))


def _is_unit_job(wf: _Workflow, job: str) -> bool:
    body = wf.body(job)
    if _runs_ingest(body) or any(_runs_ingest(wf.body(n)) for n in wf.needs_closure(job)):
        return False
    return any(_UNIT_EXPR.fullmatch(e) for e in _unit_marker_exprs(body))


def _unit_jobs() -> list[tuple[str, str]]:
    return [(f, j) for f, wf in _workflows().items() for j in wf.jobs if _is_unit_job(wf, j)]


def _pushes_image(body: str) -> bool:
    vals = re.findall(r"^\s*push:\s*(.+)$", body, re.MULTILINE)
    return any(v.strip().strip("'\"").lower() != "false" for v in vals) or "docker push" in body


def _gating_jobs(wf: _Workflow, all_wfs: dict[str, _Workflow]) -> set[str]:
    gates = set()
    for j in wf.jobs:
        if _is_unit_job(wf, j):
            gates.add(j)
        uses = wf.job_key(j, "uses")
        if uses:
            target = uses[0].strip("'\"").removeprefix("./.github/workflows/")
            called = all_wfs.get(target)
            if called and "workflow_call" in called.triggers and any(
                    _is_unit_job(called, cj) for cj in called.jobs):
                gates.add(j)
    return gates


# ── minimal Dockerfile reader ──────────────────────────────────────────────────

def _instructions(path: Path | None = None) -> list[tuple[str, str]]:
    """Logical instructions as (KEYWORD, args), continuation lines joined, comments dropped."""
    path = path or ROOT / "Dockerfile"
    out, buf = [], ""
    for ln in path.read_text().splitlines():
        if not buf and (not ln.strip() or ln.lstrip().startswith("#")):
            continue
        if buf and ln.lstrip().startswith("#"):
            continue
        buf += " " + ln.strip()
        if buf.endswith("\\"):
            buf = buf[:-1]
            continue
        kw, _, args = buf.strip().partition(" ")
        out.append((kw.upper(), " ".join(args.split())))
        buf = ""
    return out


def _final_stage() -> list[tuple[str, str]]:
    ins = _instructions()
    last_from = max(i for i, (k, _) in enumerate(ins) if k == "FROM")
    return ins[last_from:]


def _index(stage, kw: str, contains: str = "", last: bool = False) -> int | None:
    hits = [i for i, (k, a) in enumerate(stage) if k == kw and contains in a]
    return (hits[-1] if last else hits[0]) if hits else None


def _exposed_port() -> str:
    i = _index(_final_stage(), "EXPOSE")
    assert i is not None, "Dockerfile has no EXPOSE"
    return _final_stage()[i][1].split()[0].split("/")[0]


def _seed_run(stage) -> str | None:
    for k, a in stage:
        if k == "RUN" and "nfl-mcp ingest" in a and "/seed/nflread.duckdb" in a:
            return a
    return None


# ── reader self-test (hand-written oracle) ─────────────────────────────────────

def test_workflow_reader_parses_triggers_jobs_and_needs():
    sample = (
        "name: x\n"
        "on:\n  push:\n    branches: [main]\n  pull_request:\n  schedule:\n"
        "    - cron: \"0 8 * * 4\"\n"
        "jobs:\n"
        "  unit:\n    runs-on: ubuntu-latest\n    steps:\n"
        "      - run: python -m pytest -q -m 'unit'\n"
        "  slow:\n    needs: unit\n    steps:\n      - run: nfl-mcp ingest --start 2024\n"
        "  publish:\n    needs: [slow]\n    steps:\n      - uses: docker/build-push-action@v7\n"
        "        with:\n          push: true\n"
    )
    wf = _Workflow(sample)
    # Hand count: 3 triggers, 3 jobs; publish needs slow which needs unit.
    assert wf.triggers == {"push", "pull_request", "schedule"}
    assert list(wf.jobs) == ["unit", "slow", "publish"]
    assert wf.needs("publish") == ["slow"] and wf.needs_closure("publish") == {"slow", "unit"}
    assert wf.crons() == ["0 8 * * 4"]
    assert _unit_marker_exprs(wf.body("unit")) == ["unit"]
    assert _is_unit_job(wf, "unit") and not _is_unit_job(wf, "slow")
    assert _pushes_image(wf.body("publish")) and not _pushes_image(wf.body("unit"))
    assert _gating_jobs(wf, {}) == {"unit"}


# ── 1. unit-test job on push and pull_request, no network ingest ───────────────

def test_unit_job_runs_pytest_with_unit_marker():
    # Today ci.yml runs `pytest -q --cov-fail-under=100` (no -m) after an ingest step.
    assert _unit_jobs(), "no workflow job runs pytest -m unit without an ingest step"


def test_unit_job_workflow_triggers_on_push_and_pull_request():
    wfs = _workflows()
    hits = [(f, j) for f, j in _unit_jobs() if {"push", "pull_request"} <= wfs[f].triggers]
    assert hits, "no unit-test job lives in a workflow triggered by both push and pull_request"


@pytest.mark.parametrize("wf_name", ["ci.yml"])
def test_ci_has_unit_job_with_no_ingest_in_job_or_needs(wf_name):
    wf = _workflows()[wf_name]
    unit = [j for j in wf.jobs if any(_UNIT_EXPR.fullmatch(e) for e in _unit_marker_exprs(wf.body(j)))]
    assert unit, f"{wf_name} has no job running pytest -m unit"
    for j in unit:
        assert not _runs_ingest(wf.body(j)), f"{wf_name}:{j} runs nfl-mcp ingest"
        for n in wf.needs_closure(j):
            assert not _runs_ingest(wf.body(n)), f"{wf_name}:{j} needs {n}, which ingests"


def test_ci_still_triggers_on_push_and_pull_request():
    assert {"push", "pull_request"} <= _workflows()["ci.yml"].triggers


# ── 2. image publish depends on the unit-test job ──────────────────────────────

def test_docker_publish_job_needs_unit_test_job():
    wfs = _workflows()
    wf = wfs["docker.yml"]
    pushers = [j for j in wf.jobs if _pushes_image(wf.body(j))]
    assert pushers, "docker.yml has no image-pushing job"  # build-and-push today
    gates = _gating_jobs(wf, wfs)
    for j in pushers:
        assert ({j} | wf.needs_closure(j)) & gates, f"docker.yml:{j} publishes without unit tests"


@pytest.mark.parametrize("wf_name", ["docker-build.yml"])
def test_other_image_workflows_do_not_push_ungated(wf_name):
    # docker-build.yml builds with push: false today. If it ever pushes, it must be gated.
    wfs = _workflows()
    wf = wfs[wf_name]
    gates = _gating_jobs(wf, wfs)
    for j in wf.jobs:
        if _pushes_image(wf.body(j)):
            assert ({j} | wf.needs_closure(j)) & gates, f"{wf_name}:{j} pushes without unit tests"


def test_docker_publish_keeps_thursday_cron_unchanged():
    wf = _workflows()["docker.yml"]
    assert "schedule" in wf.triggers
    assert wf.crons() == ["0 8 * * 4"]


def test_docker_publish_keeps_release_published_trigger():
    wf = _workflows()["docker.yml"]
    assert "release" in wf.triggers
    rel = _children(wf.on["release"][1]).get("types")
    assert rel and "published" in _list_value(*rel)


# ── 3. HEALTHCHECK on /healthz ─────────────────────────────────────────────────

def _healthcheck() -> str:
    stage = _final_stage()
    hits = [a for k, a in stage if k == "HEALTHCHECK"]
    assert len(hits) == 1, f"expected one HEALTHCHECK in the final stage, found {len(hits)}"
    return hits[0]


def test_healthcheck_probes_healthz_with_curl_or_urllib():
    hc = _healthcheck()
    assert " CMD " in f" {hc} " and "/healthz" in hc
    assert "curl" in hc or ("python" in hc and "urllib" in hc), hc


def test_healthcheck_sets_interval_and_start_period():
    opts = _healthcheck().split(" CMD ")[0]
    assert re.search(r"--interval=\d+[smh]", opts), opts
    assert re.search(r"--start-period=\d+[smh]", opts), opts


def test_healthcheck_in_final_stage_on_exposed_port_with_curl_kept():
    stage = _final_stage()
    hc_i = _index(stage, "HEALTHCHECK")
    assert hc_i is not None and hc_i > 0  # index 0 is the last FROM
    hc = stage[hc_i][1]
    m = re.search(r"(?:localhost|127\.0\.0\.1):(\d+)/healthz", hc)
    assert m and m.group(1) == _exposed_port(), hc
    if "curl" in hc:
        runs = " ".join(a for k, a in stage if k == "RUN")
        assert not re.search(r"apt-get (purge|remove|autoremove)[^&]*curl", runs)


# ── 4. non-root USER with writable runtime dirs ────────────────────────────────

def _entrypoint_text() -> str:
    return (ROOT / "entrypoint.sh").read_text()


def _runtime_user() -> tuple[int, str]:
    """(index of ENTRYPOINT/CMD in the final stage, user entrypoint.sh drops to with setpriv)."""
    stage = _final_stage()
    entry = min(i for i in (_index(stage, "ENTRYPOINT"), _index(stage, "CMD")) if i is not None)
    m = re.search(r"setpriv\s+--reuid=(\w+)", _entrypoint_text())
    assert m, "entrypoint.sh does not drop privileges with setpriv"
    return entry, m.group(1)


def test_required_writable_dirs_match_the_container_scripts():
    # Anchors REQUIRED_WRITABLE_DIRS to what the scripts write (not to the Dockerfile).
    ep, up = (ROOT / "entrypoint.sh").read_text(), (ROOT / "update_db.sh").read_text()
    db_dirs = set(re.findall(r'^DB_DIR="([^"]+)"', ep, re.MULTILINE) + re.findall(r'^DB_DIR="([^"]+)"', up, re.MULTILINE))
    log_dirs = {str(Path(p).parent) for p in re.findall(r">>\s*(/\S+)", ep + up)}
    assert db_dirs | log_dirs == set(REQUIRED_WRITABLE_DIRS)


def test_entrypoint_drops_to_non_root_user_with_setpriv_before_writes():
    # Design change by the reviewer: the image keeps root as its start user so entrypoint.sh
    # can chown a root-owned /data mount, then re-executes itself as nflmcp.
    ep = _entrypoint_text()
    _, user = _runtime_user()
    assert user not in {"root", "0"}
    assert not any(k == "USER" for k, _ in _final_stage()), "USER would skip the chown step"
    guard = ep.index('if [ "$(id -u)" = 0 ]')
    chown = ep.index("chown -R", guard)
    drop = ep.index("setpriv --reuid=", guard)
    assert "--regid=" in ep[drop:drop + 80] and "--init-groups" in ep[drop:drop + 120]
    assert '"$0" "$@"' in ep[drop:drop + 200]
    assert guard < chown < drop
    # Every write to /data, the cron file and the server start come after the drop.
    for write in ('cp "$SEED"', 'mv "$SYMLINK"', 'ln -sfn', "nfl-mcp ingest", "cat > /tmp/crontab",
                  "supercronic /tmp/crontab", "exec nfl-mcp serve"):
        assert drop < ep.index(write), write


@pytest.mark.parametrize("directory", REQUIRED_WRITABLE_DIRS)
def test_runtime_dir_made_writable_before_user_switch(directory):
    idx, user = _runtime_user()
    name = re.escape(user.split(":")[0])
    d = re.escape(directory)
    runs = [a for k, a in _final_stage()[:idx] if k == "RUN"]
    chown = re.compile(rf"chown\s+(-R\s+)?{name}(:\S+)?\s+[^&;]*{d}(\s|$)")
    chmod = re.compile(rf"chmod\s+(-R\s+)?(1?7[77][07]|[aogu]+\+w\S*)\s+[^&;]*{d}(\s|$)")
    assert any(chown.search(r) or chmod.search(r) for r in runs), directory


def test_runtime_user_is_created_and_seed_bake_runs_before_it():
    idx, user = _runtime_user()
    stage = _final_stage()
    name = user.split(":")[0]
    runs_before = " ".join(a for k, a in stage[:idx] if k == "RUN")
    if not name.isdigit():
        assert re.search(rf"(useradd|adduser)\b[^&;]*\b{re.escape(name)}\b", runs_before), name
    seed_i = next((i for i, (k, a) in enumerate(stage) if k == "RUN" and a == _seed_run(stage)), None)
    if seed_i is not None:
        assert seed_i < idx or re.search(r"(chown|chmod)[^&;]*/seed", runs_before)


# ── 5. bind defaults (regression) ──────────────────────────────────────────────

def test_serve_host_option_defaults_to_loopback():
    host = next(p for p in cli.serve.params if p.name == "host")
    assert host.default == "127.0.0.1"


def test_serve_help_shows_loopback_default():
    result = CliRunner().invoke(cli.main, ["serve", "--help"])
    assert result.exit_code == 0
    assert re.search(r"--host.*default:\s*127\.0\.0\.1", " ".join(result.output.split()))


def test_entrypoint_serves_all_interfaces_on_exposed_port():
    lines = [ln for ln in (ROOT / "entrypoint.sh").read_text().splitlines()
             if ln.strip().startswith("exec nfl-mcp")]   # not the exec setpriv re-run
    assert len(lines) == 1
    argv = shlex.split(lines[0])
    assert argv[1:3] == ["nfl-mcp", "serve"]
    assert argv[argv.index("--host") + 1] == "0.0.0.0"
    assert argv[argv.index("--port") + 1] == _exposed_port()


# ── 6. seed bake and check (regression; complements tests/test_dockerfile_dev.py) ─

def test_seed_bake_is_in_the_final_stage():
    stage = _final_stage()
    copied = any(k == "COPY" and "--from=" in a and "/seed" in a for k, a in stage)
    assert _seed_run(stage) is not None or copied


def test_seed_ingest_and_checks_are_one_chained_run_in_order():
    run = _seed_run(_final_stage()) or _seed_run(_instructions())
    assert run is not None
    i_ingest = run.index("nfl-mcp ingest")
    i_wal = run.index("test ! -e /seed/nflread.duckdb.wal")
    i_rows = run.index("('plays','schedules')")
    assert i_ingest < i_wal < i_rows
    # Chained with && (no ';' or '|| true'), so any failing step fails the build.
    ingest_to_wal = run[i_ingest:i_wal]
    assert "&&" in ingest_to_wal and ";" not in ingest_to_wal
    assert "&& python -c" in run[i_wal:i_rows]
    assert "|| true" not in run


def test_seed_row_check_opens_read_only_and_fails_on_empty_tables():
    run = _seed_run(_final_stage()) or _seed_run(_instructions())
    assert run is not None
    assert "duckdb.connect('/seed/nflread.duckdb',read_only=True)" in run
    assert "sys.exit(0 if all(v>0 for v in rows.values()) else 1)" in run
