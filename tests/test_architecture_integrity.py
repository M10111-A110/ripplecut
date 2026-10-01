"""Master spec §122 / §97-§100: architectural integrity, checked statically on the source tree.

Exactly one dependency engine, cascade simulator, intervention engine, objective evaluator and safety
validator; many solvers; solvers own none of: model, dependency semantics, objective, validation rules,
UI, LLM logic. The decision path never names a specific algorithm (Solver Independence Invariant).
"""
import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "ripplecut"
ALL = sorted(PKG.rglob("*.py"))
# Solver-framework modules that legitimately touch the engine (the context) or name solvers (registration).
FRAMEWORK = {"base.py", "registry.py", "policy.py", "guard.py", "orchestrator.py", "builtin.py", "fault_injection.py",
             "__init__.py"}
SOLVER_MODULES = [p for p in (PKG / "solvers").glob("*.py") if p.name not in FRAMEWORK]


def _defs(name_kind):
    out = []
    for p in ALL:
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name_kind:
                out.append(p.relative_to(PKG).as_posix())
    return out


def test_exactly_one_of_each_core_component():
    assert _defs("phi") == ["engine/simulator.py"]
    assert _defs("simulate") == ["engine/simulator.py"]
    assert _defs("compiled_rules") == ["engine/dependency.py"]
    assert _defs("check_plan") == ["engine/interventions.py"]
    assert _defs("evaluate_plan") == ["engine/objective.py"]
    assert _defs("compute_objective") == ["engine/objective.py"]
    assert _defs("SafetyValidator") == ["validation/validator.py"]


def test_there_are_several_solvers():
    assert {p.name for p in SOLVER_MODULES} >= {"branch_and_bound.py", "exhaustive.py"}


def _imports_and_calls(path):
    tree = ast.parse(path.read_text())
    imports, calls = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
            imports.update(f"{node.module}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Import):
            imports.update(a.name for a in node.names)
        elif isinstance(node, ast.Call):
            f = node.func
            calls.add(f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else "")
    return imports, calls


def test_solvers_do_not_own_semantics_objective_validation_ui_or_llm():
    forbidden_imports = ("engine.simulator", "engine.dependency", "engine.interventions", "validation", "incident",
                         "explain", "ui", "pipeline", "evidence", "model.loader", "model.validation")
    forbidden_calls = {"simulate", "phi", "evaluate_plan", "check_plan", "compute_objective", "rule_values",
                       "apply_plan", "transform", "load_model", "parse_model"}
    for p in SOLVER_MODULES:
        imports, calls = _imports_and_calls(p)
        bad_imports = [i for i in imports if any(i.lstrip(".").startswith(f) for f in forbidden_imports)]
        assert not bad_imports, (p.name, bad_imports)
        assert not (calls & forbidden_calls), (p.name, calls & forbidden_calls)
        text = p.read_text()
        assert "def objective_key" not in text and "rule_type" not in text


def test_decision_path_never_names_an_algorithm():
    names = ("branch_and_bound", "branchandbound", "exhaustive")
    checked = [PKG / "pipeline.py", PKG / "solvers" / "orchestrator.py", PKG / "solvers" / "guard.py",
               PKG / "solvers" / "registry.py", PKG / "solvers" / "policy.py", PKG / "solvers" / "base.py",
               PKG / "ui" / "server.py", PKG / "ui" / "static" / "index.html"]
    checked += list((PKG / "validation").glob("*.py")) + list((PKG / "engine").glob("*.py"))
    checked += list((PKG / "explain").glob("*.py")) + list((PKG / "incident").glob("*.py"))
    checked += list((PKG / "evidence").glob("*.py")) + list((PKG / "model").glob("*.py"))
    for p in checked:
        low = p.read_text().lower()
        assert not any(n in low for n in names), p


def test_objective_is_lexicographic_only():
    for p in ALL:
        text = p.read_text()
        assert "weighted_score" not in text and "alpha *" not in text, p


def test_llm_never_imported_by_core():
    core = list((PKG / "engine").glob("*.py")) + list((PKG / "solvers").glob("*.py")) + \
        list((PKG / "validation").glob("*.py")) + list((PKG / "model").glob("*.py"))
    for p in core:
        imports, _ = _imports_and_calls(p)
        assert not any("llm" in i or "urllib" in i for i in imports), p
