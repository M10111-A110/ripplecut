"""Evidence sources for the data adapter (master spec §49-§51, §104-§105).

    raw evidence -> preprocessing -> service observations -> StateEstimator -> RippleCut state

This layer is independent from the deterministic core: it only produces an
initial state x(0). It never produces a containment recommendation.

* ``ReplayFixtureSource`` loads a local JSON replay fixture (offline demo).
  Every fixture must declare its provenance; the shipped fixture is labeled
  SYNTHETIC and is not attributed to RCAEval.
* ``RCAEvalCaseSource`` is the adapter boundary for a real RCAEval case
  directory. In this build, these are synthetic-derived fixtures, not real inspected RCAEval data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from ..errors import DataAdapterError
from ..model.schema import SystemModel

ALLOWED_PROVENANCE = ("SYNTHETIC", "RCAEVAL_INSPECTED", "SYNTHETIC_DERIVED", "OFFICIAL_RCAEVAL")
# Per the RCAEval README at commit 259ea41 (layout only; file *contents* were not inspected):
# case dirs such as re1ob_adservice_cpu_1 hold metrics.json (Figshare/Zenodo) or metrics.parquet (Hugging Face)
# and inject_time.txt (Unix timestamp of the fault injection).
RCAEVAL_EXPECTED_FILES = ("metrics.json", "metrics.parquet", "inject_time.txt", "metadata.json")


@dataclass(frozen=True)
class EvidenceBundle:
    source_id: str
    provenance: Mapping[str, Any]
    observations: Mapping[str, Any]
    path: Optional[str] = None

    @property
    def label(self) -> str:
        return str(self.provenance.get("label", self.provenance.get("kind", "UNLABELED")))

    def to_dict(self) -> Dict[str, Any]:
        return {"source_id": self.source_id, "provenance": dict(self.provenance), "path": self.path}


class ReplayFixtureSource:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> EvidenceBundle:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise DataAdapterError(f"replay fixture not found: {self.path}") from None
        except json.JSONDecodeError as exc:
            raise DataAdapterError(f"replay fixture is not valid JSON: {exc}") from None
        if not isinstance(raw, Mapping):
            raise DataAdapterError("replay fixture must be a JSON object")
        prov = raw.get("provenance")
        if not isinstance(prov, Mapping) or prov.get("kind") not in ALLOWED_PROVENANCE:
            raise DataAdapterError(f"replay fixture must declare provenance.kind in {ALLOWED_PROVENANCE}")
        if prov.get("kind") == "RCAEVAL_INSPECTED" and not prov.get("case_id"):
            raise DataAdapterError("an RCAEval-derived fixture must name the inspected case_id")
        if not isinstance(raw.get("observations"), Mapping):
            raise DataAdapterError("replay fixture needs an 'observations' object")
        return EvidenceBundle(source_id=str(raw.get("fixture_id", self.path.stem)), provenance=dict(prov),
                              observations=raw["observations"], path=str(self.path))


class RCAEvalCaseSource:
    """Adapter boundary for one RCAEval case directory ({benchmark}_{service}_{fault}_{instance})."""

    SCHEMA_STATUS = "SYNTHETIC_DERIVED_RE1_OB"

    def __init__(self, case_dir: Path, system: Optional[SystemModel] = None,
                 estimator_config: Optional[Mapping[str, Any]] = None) -> None:
        self.case_dir = Path(case_dir)
        self.system = system
        self.estimator_config = estimator_config

    def check_layout(self) -> Dict[str, bool]:
        if not self.case_dir.is_dir():
            raise DataAdapterError(f"RCAEval case directory not found: {self.case_dir}")
        return {name: (self.case_dir / name).exists() for name in RCAEVAL_EXPECTED_FILES}

    def load(self, system: Optional[SystemModel] = None) -> EvidenceBundle:
        layout = self.check_layout()
        sys = system or self.system
        if sys is None:
            raise DataAdapterError("RCAEval metric-to-observation mapping REQUIRES VERIFICATION: "
                                   "RCAEvalCaseSource requires a SystemModel to map services",
                                   details={"case_dir": str(self.case_dir)})

        from .rcaeval_adapter import RCAEvalAdapter
        adapter = RCAEvalAdapter(sys, self.estimator_config or {})
        case_data = adapter.load_case(self.case_dir)
        obs_dict = {
            sid: o.to_partial_observation_dict() if o.missing_signals else o.to_observation_dict()
            for sid, o in case_data.normalized_observations.items()
        }

        data_status = getattr(case_data.metadata, "data_status", getattr(case_data.metadata, "source", "RCAEVAL_INSPECTED"))

        prov = {
            "kind": data_status,
            "case_id": case_data.metadata.case_id,
            "benchmark": case_data.metadata.benchmark,
            "ground_truth_service": case_data.metadata.ground_truth_service,
            "fault_type": case_data.metadata.fault_type,
            "instance": case_data.metadata.instance,
            "inject_time": case_data.metadata.inject_time,
            "label": f"RCAEval Inspected Case: {case_data.metadata.case_id}",
        }

        return EvidenceBundle(
            source_id=case_data.metadata.case_id,
            provenance=prov,
            observations=obs_dict,
            path=str(self.case_dir),
        )

