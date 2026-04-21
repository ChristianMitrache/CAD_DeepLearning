"""Parse AutoMate assembly JSON files into typed dataclasses.

Schema reference: data/automate/README.md section "Assembly JSONS".
All distances are in meters. Transforms are 4x4 row-major homogeneous.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Part:
    id: str
    has_parasolid: bool
    has_step: bool

    def step_path(self, data_dir: Path) -> Path:
        # Zenodo zips extract with a redundant top-level dir -> step/step/, assemblies/assemblies/
        return data_dir / "step" / "step" / f"{self.id}.step"


@dataclass
class Occurrence:
    index: int  # position in the occurrences list
    part_index: int
    id: str
    transform: np.ndarray  # 4x4 float
    fixed: bool
    hidden: bool
    has_parasolid: bool
    has_step: bool


@dataclass
class Mate:
    name: str
    id: str
    mate_type: str
    occurrence_indices: list[int]
    mcfs: list[np.ndarray]  # each 4x4
    has_parasolid: bool
    has_step: bool


@dataclass
class AutoMateAssembly:
    id: str
    has_all_parasolid: bool
    has_all_step: bool
    parts: list[Part]
    occurrences: list[Occurrence]
    mates: list[Mate]
    raw: dict = field(repr=False)  # keep original JSON for mateRelations etc

    # ------------------------------------------------------------------ loaders
    @classmethod
    def from_json(cls, assembly_id: str, data_dir: Path) -> AutoMateAssembly:
        # Zenodo zips extract with a redundant top-level dir -> assemblies/assemblies/
        json_path = Path(data_dir) / "assemblies" / "assemblies" / f"{assembly_id}.json"
        with json_path.open() as f:
            doc = json.load(f)
        return cls.from_dict(doc)

    @classmethod
    def from_dict(cls, doc: dict) -> AutoMateAssembly:
        parts = [
            Part(
                id=p["id"],
                has_parasolid=p.get("has_parasolid", False),
                has_step=p.get("has_step", False),
            )
            for p in doc.get("parts", [])
        ]
        occurrences = [
            Occurrence(
                index=i,
                part_index=o["part"],
                id=o["id"],
                transform=np.asarray(o["transform"], dtype=np.float64).reshape(4, 4),
                fixed=o.get("fixed", False),
                hidden=o.get("hidden", False),
                has_parasolid=o.get("has_parasolid", False),
                has_step=o.get("has_step", False),
            )
            for i, o in enumerate(doc.get("occurrences", []))
        ]
        mates = [
            Mate(
                name=m.get("name", ""),
                id=m["id"],
                mate_type=m["mateType"],
                occurrence_indices=list(m.get("occurrences", [])),
                mcfs=[np.asarray(f, dtype=np.float64).reshape(4, 4) for f in m.get("mcfs", [])],
                has_parasolid=m.get("has_parasolid", False),
                has_step=m.get("has_step", False),
            )
            for m in doc.get("mates", [])
        ]
        return cls(
            id=doc.get("assemblyId", ""),
            has_all_parasolid=doc.get("has_all_parasolid", False),
            has_all_step=doc.get("has_all_step", False),
            parts=parts,
            occurrences=occurrences,
            mates=mates,
            raw=doc,
        )

    # ------------------------------------------------------------------ helpers
    def visible_occurrences(self) -> list[Occurrence]:
        return [o for o in self.occurrences if not o.hidden]

    def step_occurrences(self, data_dir: Path) -> list[tuple[Occurrence, Path]]:
        """Yield (occurrence, step_path) for occurrences whose part has a step file."""
        data_dir = Path(data_dir)
        out = []
        for o in self.occurrences:
            if not o.has_step or o.hidden:
                continue
            path = self.parts[o.part_index].step_path(data_dir)
            out.append((o, path))
        return out

    def mate_type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for m in self.mates:
            counts[m.mate_type] = counts.get(m.mate_type, 0) + 1
        return counts

    def summary(self) -> dict:
        return {
            "id": self.id,
            "n_parts": len(self.parts),
            "n_occurrences": len(self.occurrences),
            "n_occurrences_visible": len(self.visible_occurrences()),
            "n_mates": len(self.mates),
            "mate_types": self.mate_type_counts(),
            "has_all_step": self.has_all_step,
        }
