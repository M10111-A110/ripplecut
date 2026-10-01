"""RippleCut command-line interface (offline by default; master spec §104-§106).

    python -m ripplecut demo                     all canonical scenarios + resilience test
    python -m ripplecut run --scenario payment_failure
    python -m ripplecut run --text "paymentservice is down"
    python -m ripplecut run --failed paymentservice,shippingservice
    python -m ripplecut run --fixture config/fixtures/synthetic_observed_checkout_down.json
    python -m ripplecut run --scenario payment_failure --fault timeout     (resilience test)
    python -m ripplecut scenarios | validate-config | benchmark | serve
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .errors import RippleCutError
from .logs import configure_cli_logging
from .pipeline import FaultSpec, RippleCutApp, validate_config
from .solvers.fault_injection import FAULT_MODES

EXIT_OK, EXIT_INPUT, EXIT_NO_VALID, EXIT_MISMATCH = 0, 2, 3, 4


def _fmt_obj(o: Optional[Dict[str, Any]]) -> str:
    return f"(K={o['K']}, N={o['N']}, R={o['R']}; C={o['C']}/{o['C_min']})" if o else "-"


def render_report(r: Dict[str, Any]) -> str:
    out: List[str] = []
    bar = "=" * 78
    out.append(bar)
    title = (r.get("scenario") or {}).get("title") or "ad-hoc incident"
    out.append(f"RippleCut | {r['label']} | {title}")
    if r.get("resilience_test"):
        rt = r["resilience_test"]
        out.append(f"*** {rt['label']}: forcing '{rt['mode']}' in solver '{rt['target_solver']}' ***")
    out.append(bar)
    out.append("[1] INCIDENT")
    inp = r["input"]
    if inp["kind"] == "text":
        out.append(f"    natural language : {inp['text']!r}")
        p = r.get("parse") or {}
        out.append(f"    parser           : {p.get('parser')} -> {p.get('status')}")
        if p.get("status") != "OK":
            out.append(f"    message          : {p.get('message')}")
            for alt in p.get("alternatives", []):
                out.append(f"      option '{alt['id']}': {alt['label']}  -> --failed "
                           f"{','.join(alt['incident']['failed_services'])}")
    elif inp["kind"] == "replay_fixture":
        ev = r.get("evidence") or {}
        out.append(f"    replay fixture   : {inp['path']}")
        if ev:
            out.append(f"    provenance       : {ev['provenance'].get('label')}")
            for s in ev["estimated_state"]["services"]:
                if s["state"] != "UP":
                    out.append(f"      {s['service']:<24} {s['state']:<9} (x=0)  {s['reason']}")
    if r.get("incident"):
        out.append(f"    structured       : {json.dumps({k: v for k, v in r['incident'].items() if k in ('failed_services', 'degraded_services', 'scenario_id')})}")
    if r.get("error"):
        out.append(f"    STATUS           : {r['status']}: {r['error'].get('message')}")
        return "\n".join(out)
    uc = r["uncontrolled_cascade"]
    out.append("[2] UNCONTROLLED CASCADE  (Phi(x)_v = x_v AND R_v(x); RIPPLECUT-MODELED rules)")
    for rnd in uc["rounds"]:
        if rnd["round"] == 0:
            out.append(f"    round 0 (initial): DOWN {rnd['down']}")
        else:
            out.append(f"    round {rnd['round']}          : newly failed {rnd['newly_failed']}")
    out.append(f"    fixed point after {uc['state_changing_rounds']} state-changing round(s), "
               f"{uc['phi_evaluations']} Phi evaluations: DOWN {uc['final_down']}")
    pl = r["planner"]
    out.append(f"[3] PLANNER  m={r['problem']['m']} actions, 2^m={r['problem']['candidates']} candidate plans, "
               f"order={pl['policy']['execution_order']}")
    for a in pl["attempts"]:
        res = a["result"]
        out.append(f"    {a['solver']:<22} {res['status']:<17} -> {a['outcome']:<22} "
                   f"{res['runtime_seconds']*1000:8.2f} ms" + (f"  ({res['error']})" if res.get("error") and
                                                                a['outcome'] != "ACCEPTED" else ""))
    out.append(f"    verification     : {pl['verification'].get('status')}")
    res = r["result"]
    out.append("[4] RESULT")
    out.append(f"    status           : {r['status']}")
    if r["status"] == "SUCCESS":
        out.append(f"    plan             : {res['plan'] or '[] (do nothing)'}")
        out.append(f"    objective        : {_fmt_obj(res['objective'])}")
    elif r["status"] == "NO_FEASIBLE_PLAN":
        out.append(f"    infeasibility    : certified={res.get('infeasibility_certified')}")
    out.append(f"    optimality       : {res['optimality_status']}")
    out.append(f"    validation       : {res['validation_status']}  (solver={res['selected_solver']}, "
               f"fallbacks={res['fallback_count']})")
    out.append(f"    approval         : {r['approval']['status']}  ({r['approval']['note']})")
    out.append("[5] WHY")
    for line in r["explanation"]["text"].splitlines():
        out.append("    " + line)
    return "\n".join(out)


def _exit_code(r: Dict[str, Any]) -> int:
    if r["status"] in ("SUCCESS", "NO_FEASIBLE_PLAN"):
        return EXIT_OK
    if r["status"] == "NO_VALID_SOLVER_RESULT":
        return EXIT_NO_VALID
    return EXIT_INPUT


def check_expectation(scn: Dict[str, Any], r: Dict[str, Any]) -> List[str]:
    exp = scn.get("expected") or {}
    problems = []
    if exp.get("status") and r["status"] != exp["status"]:
        problems.append(f"status {r['status']} != {exp['status']}")
    if "uncontrolled_down" in exp and r.get("uncontrolled_cascade", {}).get("final_down") != exp["uncontrolled_down"]:
        problems.append("uncontrolled cascade differs")
    if exp.get("status") == "SUCCESS":
        res = r.get("result") or {}
        if res.get("plan") != exp.get("plan"):
            problems.append(f"plan {res.get('plan')} != {exp.get('plan')}")
        obj = res.get("objective") or {}
        for k in ("K", "N", "R", "C"):
            if k in exp and obj.get(k) != exp[k]:
                problems.append(f"{k} {obj.get(k)} != {exp[k]}")
    if "infeasibility_certified" in exp and (r.get("result") or {}).get("infeasibility_certified") != \
            exp["infeasibility_certified"]:
        problems.append("infeasibility certification differs")
    return problems


def cmd_demo(app: RippleCutApp, args: argparse.Namespace) -> int:
    rows, reports, mismatches = [], [], 0
    for s in app.scenarios:
        r = app.run_scenario(s["id"])
        reports.append(r)
        probs = check_expectation(s, r)
        mismatches += bool(probs)
        res = r["result"]
        rows.append((s["id"], ",".join(r["uncontrolled_cascade"]["final_down"]), r["status"],
                     ",".join(res.get("plan") or []) if res.get("plan") is not None else "-",
                     _fmt_obj(res.get("objective")) if res.get("objective") else "-",
                     res["optimality_status"] if r["status"] != "NO_FEASIBLE_PLAN" else
                     ("CERTIFIED INFEASIBLE" if res.get("infeasibility_certified") else "INFEASIBLE (uncertified)"),
                     "MATCH" if not probs else "MISMATCH: " + "; ".join(probs)))
    fault = FaultSpec(args.fault or "timeout", timeout_seconds=args.fault_timeout)
    rt = app.run_scenario("payment_failure", fault=fault) if any(s["id"] == "payment_failure" for s in app.scenarios) \
        else app.run_scenario(app.scenarios[0]["id"], fault=fault)
    if args.json:
        print(json.dumps({"scenarios": reports, "resilience_test": rt}, indent=2, default=str))
        return EXIT_MISMATCH if mismatches else EXIT_OK
    print("RippleCut demo — ONE Online Boutique topology, ONE rule set, ONE action space; different incidents.")
    print("All scenarios are Controlled RippleCut Scenarios (no RCAEval case was inspected in this build).\n")
    w = [29, 70, 17, 41, 24, 21]
    head = ("scenario", "uncontrolled fixed point: DOWN", "status", "recommended plan", "objective", "optimality")
    print("  ".join(h.ljust(x) for h, x in zip(head, w)) + "  expected")
    for row in rows:
        print("  ".join(str(c).ljust(x) for c, x in zip(row[:-1], w)) + "  " + row[-1])
    print()
    print(render_report(reports[0]))
    print()
    print(render_report(rt))
    print(f"\nScenario expectations: {len(rows) - mismatches}/{len(rows)} match.")
    return EXIT_MISMATCH if mismatches else EXIT_OK


def cmd_run(app: RippleCutApp, args: argparse.Namespace) -> int:
    fault = FaultSpec(args.fault, args.fault_solver, args.fault_timeout) if args.fault else None
    if args.scenario:
        r = app.run_scenario(args.scenario, via_text=args.via_text, fault=fault, use_llm=args.llm)
    elif args.text:
        r = app.run_text(args.text, fault=fault, use_llm=args.llm)
    elif args.fixture:
        r = app.run_fixture(Path(args.fixture).resolve(), fault=fault)
    else:
        failed = [s for s in (args.failed or "").split(",") if s]
        degraded = [s for s in (args.degraded or "").split(",") if s]
        inc: Dict[str, Any] = {"failed_services": failed}
        if degraded:
            inc["degraded_services"] = degraded
        r = app.run_incident(inc, fault=fault)
    print(json.dumps(r, indent=2, default=str) if args.json else render_report(r))
    return _exit_code(r)


def cmd_scenarios(app: RippleCutApp, args: argparse.Namespace) -> int:
    for s in app.scenarios:
        src = f"fixture={s['fixture']}" if "fixture" in s else f"incident={json.dumps(s['incident'])}"
        print(f"{s['id']:<30} {s.get('label', '')}\n    {s.get('title', '')}\n    {src}")
    return EXIT_OK


def cmd_validate(app: RippleCutApp, args: argparse.Namespace) -> int:
    print(json.dumps(validate_config(app), indent=2))
    return EXIT_OK


def cmd_benchmark(app: RippleCutApp, args: argparse.Namespace) -> int:
    from .benchmark import run_benchmark, write_markdown
    data = run_benchmark(app, repeats=args.repeats, seeds=args.seeds, quick=args.quick)
    if args.out:
        write_markdown(data, Path(args.out))
        print(f"wrote {args.out}")
    print(json.dumps(data["summary"], indent=2))
    return EXIT_OK if data["summary"]["all_equivalent"] else EXIT_MISMATCH


def cmd_serve(app: RippleCutApp, args: argparse.Namespace) -> int:
    from .ui.server import serve
    serve(app, host=args.host, port=args.port)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ripplecut", description="RippleCut containment planner (offline by default)")
    p.add_argument("--model", help="model config (default config/online_boutique_canonical.json)")
    p.add_argument("--policy", help="solver policy (default config/solver_policy.json)")
    p.add_argument("--scenarios", help="scenario file (default config/scenarios.json)")
    p.add_argument("--verbose", action="store_true", help="print structured JSON log events to stderr")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="run every canonical scenario plus a resilience test")
    d.add_argument("--json", action="store_true")
    d.add_argument("--fault", choices=FAULT_MODES, help="fault mode for the resilience test (default timeout)")
    d.add_argument("--fault-timeout", type=float, default=1.5)

    r = sub.add_parser("run", help="run one incident")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--scenario")
    g.add_argument("--text", help="natural-language incident (deterministic parser unless --llm)")
    g.add_argument("--failed", help="comma-separated failed service ids (structured input)")
    g.add_argument("--fixture", help="replay fixture JSON (observations -> state estimator)")
    r.add_argument("--degraded", help="comma-separated degraded service ids (with --failed)")
    r.add_argument("--via-text", action="store_true", help="with --scenario: use the scenario's example text")
    r.add_argument("--llm", action="store_true", help="use the optional LLM parser/explainer if configured")
    r.add_argument("--fault", choices=FAULT_MODES, help="Solver failure simulation / resilience test")
    r.add_argument("--fault-solver", help="solver to break (default: first in policy order)")
    r.add_argument("--fault-timeout", type=float, default=1.5)
    r.add_argument("--json", action="store_true")

    sub.add_parser("scenarios", help="list canonical scenarios")
    sub.add_parser("validate-config", help="validate model, policy and scenarios")
    b = sub.add_parser("benchmark", help="Exhaustive vs Branch & Bound (measured)")
    b.add_argument("--repeats", type=int, default=5)
    b.add_argument("--seeds", type=int, default=20)
    b.add_argument("--quick", action="store_true")
    b.add_argument("--out", help="write a Markdown report (e.g. docs/BENCHMARK.md)")
    s = sub.add_parser("serve", help="start the local demo UI (offline)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    configure_cli_logging(args.verbose)
    try:
        app = RippleCutApp(model_path=args.model, policy_path=args.policy, scenarios_path=args.scenarios,
                           llm_client=_llm_client() if getattr(args, "llm", False) else None)
    except RippleCutError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_INPUT
    handlers = {"demo": cmd_demo, "run": cmd_run, "scenarios": cmd_scenarios, "validate-config": cmd_validate,
                "benchmark": cmd_benchmark, "serve": cmd_serve}
    try:
        return handlers[args.command](app, args)
    except RippleCutError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT


def _llm_client():
    from .incident.llm import optional_client
    return optional_client()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
