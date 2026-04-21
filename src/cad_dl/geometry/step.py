"""STEP file I/O and tessellation to trimesh.

Pipeline: STEP -> OCC TopoDS_Shape -> BRepMesh_IncrementalMesh -> per-face triangles
-> trimesh.Trimesh. Dataset-agnostic; downstream callers apply their own unit scaling
and occurrence transforms.
"""
# pyright: reportArgumentType=false
# pythonocc-core's shipped stubs mis-type the enum constants (TopAbs_FACE etc.)
# as int instead of their enum types. Silence here; OCC usage is correct at runtime.
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import trimesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import topods

# AutoMate STEP parts are authored in mm; occurrence transforms are in m.
# Most CAD formats follow the same convention, so this lives here as a shared constant.
MM_TO_M = 1.0 / 1000.0


def load_step_shape(path: Path):
    """Read a STEP file into an OCC TopoDS_Shape."""
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != 1:  # IFSelect_RetDone
        raise RuntimeError(f"STEP read failed for {path} (status={status})")
    reader.TransferRoots()
    return reader.OneShape()


def shape_to_trimesh(shape, deflection: float = 0.5, angular: float = 0.5) -> trimesh.Trimesh | None:
    """Tessellate an OCC shape to a single trimesh.Trimesh.

    Returns None if the shape has no triangulable faces.
    """
    BRepMesh_IncrementalMesh(shape, deflection, False, angular, True)

    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    vert_offset = 0

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods.Face(explorer.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is None:
            explorer.Next()
            continue
        trsf = loc.Transformation()
        n_nodes = tri.NbNodes()
        n_tris = tri.NbTriangles()
        if n_nodes == 0 or n_tris == 0:
            explorer.Next()
            continue

        verts = np.empty((n_nodes, 3), dtype=np.float64)
        for i in range(1, n_nodes + 1):
            pnt = tri.Node(i).Transformed(trsf)
            verts[i - 1] = (pnt.X(), pnt.Y(), pnt.Z())

        faces = np.empty((n_tris, 3), dtype=np.int64)
        reversed_face = face.Orientation() == 1  # TopAbs_REVERSED
        for i in range(1, n_tris + 1):
            t = tri.Triangle(i)
            a, b, c = t.Value(1), t.Value(2), t.Value(3)
            if reversed_face:
                a, b = b, a
            faces[i - 1] = (a - 1 + vert_offset, b - 1 + vert_offset, c - 1 + vert_offset)

        all_verts.append(verts)
        all_faces.append(faces)
        vert_offset += n_nodes
        explorer.Next()

    if not all_verts:
        return None

    verts = np.vstack(all_verts)
    faces = np.vstack(all_faces)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def render_step_file(
    step_path: Path,
    out_path: Path,
    size: int = 512,
    deflection: float = 0.5,
    background: str = "white",
) -> bool:
    """Smoke-test helper: tessellate a single STEP and render it to PNG."""
    from cad_dl.geometry.render import vtk_render_meshes

    shape = load_step_shape(Path(step_path))
    mesh = shape_to_trimesh(shape, deflection=deflection)
    if mesh is None:
        return False
    vtk_render_meshes([(mesh, (0.6, 0.7, 0.9))], Path(out_path), size=size, background=background)
    return True
