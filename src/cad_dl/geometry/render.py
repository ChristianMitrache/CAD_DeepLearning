"""VTK offscreen rendering helpers for meshes and point clouds.

Fully tears down the render window per call to avoid Cocoa/GL context reuse issues
when called from pooled worker processes. Also includes a SIGALRM-based timeout
context manager so runaway tessellations can be killed cleanly.
"""
# pyright: reportAttributeAccessIssue=false, reportMissingImports=false
# VTK attaches every vtk.vtkXxx class dynamically via from-import loops, so
# static analyzers can't see vtkPolyData / vtkRenderer / etc. Silence here.
from __future__ import annotations

import contextlib
import os
import signal
from pathlib import Path

import numpy as np

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import trimesh
import vtk
from vtk.util import numpy_support

vtk.vtkObject.GlobalWarningDisplayOff()


class RenderTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise RenderTimeout()


@contextlib.contextmanager
def render_timeout(seconds: int):
    """SIGALRM-based timeout. No-op on platforms without SIGALRM (Windows)."""
    if not hasattr(signal, "SIGALRM") or seconds <= 0:
        yield
        return
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _trimesh_to_vtk_polydata(mesh: trimesh.Trimesh) -> vtk.vtkPolyData:
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(mesh.vertices, dtype=np.float32)))
    faces = mesh.faces.astype(np.int64)
    n = faces.shape[0]
    cells = np.empty((n, 4), dtype=np.int64)
    cells[:, 0] = 3
    cells[:, 1:] = faces
    id_arr = numpy_support.numpy_to_vtkIdTypeArray(cells.ravel(), deep=True)
    ca = vtk.vtkCellArray()
    ca.SetCells(n, id_arr)
    pd = vtk.vtkPolyData()
    pd.SetPoints(points)
    pd.SetPolys(ca)
    return pd


_BG_RGB = {"white": (1, 1, 1), "black": (0, 0, 0), "gray": (0.5, 0.5, 0.5)}


def _setup_isometric_camera(renderer, azimuth: float = 30.0, elevation: float = 30.0, zoom: float = 1.1) -> None:
    camera = renderer.GetActiveCamera()
    camera.SetPosition(1, 1, 1)
    camera.SetFocalPoint(0, 0, 0)
    camera.SetViewUp(0, 0, 1)
    renderer.ResetCamera()
    camera.Azimuth(azimuth)
    camera.Elevation(elevation)
    renderer.ResetCameraClippingRange()
    camera.Zoom(zoom)


def _write_png(render_window, out_path: Path) -> None:
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(render_window)
    w2i.SetInputBufferTypeToRGB()
    w2i.ReadFrontBufferOff()
    w2i.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(out_path))
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()


def vtk_render_meshes(
    meshes: list[tuple[trimesh.Trimesh, tuple[float, float, float]]],
    out_path: Path,
    size: int = 512,
    background: str = "white",
) -> None:
    """Off-screen render a list of (mesh, rgb) to PNG."""
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(*_BG_RGB.get(background, (1, 1, 1)))

    for mesh, color in meshes:
        pd = _trimesh_to_vtk_polydata(mesh)
        norms = vtk.vtkPolyDataNormals()
        norms.SetInputData(pd)
        norms.SetFeatureAngle(30.0)
        norms.SplittingOff()
        norms.ConsistencyOn()
        norms.AutoOrientNormalsOn()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(norms.GetOutputPort())
        mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetInterpolationToPhong()
        renderer.AddActor(actor)

    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.SetSize(size, size)
    render_window.AddRenderer(renderer)

    _setup_isometric_camera(renderer, azimuth=30, elevation=30)
    render_window.Render()
    _write_png(render_window, out_path)
    render_window.Finalize()


def vtk_render_points(
    clouds: list[tuple[np.ndarray, tuple[float, float, float]]],
    out_path: Path,
    size: int = 512,
    background: str = "white",
    point_size: float = 2.5,
) -> None:
    """Off-screen render a list of (Nx3 points, rgb) as one merged colored point cloud."""
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(*_BG_RGB.get(background, (1, 1, 1)))

    total = sum(len(c) for c, _ in clouds)
    all_pts = np.empty((total, 3), dtype=np.float32)
    all_rgb = np.empty((total, 3), dtype=np.uint8)
    off = 0
    for pts, color in clouds:
        n = len(pts)
        all_pts[off:off + n] = pts
        all_rgb[off:off + n] = np.array(
            [int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)], dtype=np.uint8
        )
        off += n

    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(all_pts), deep=True))

    verts = np.empty((total, 2), dtype=np.int64)
    verts[:, 0] = 1
    verts[:, 1] = np.arange(total, dtype=np.int64)
    id_arr = numpy_support.numpy_to_vtkIdTypeArray(verts.ravel(), deep=True)
    ca = vtk.vtkCellArray()
    ca.SetCells(total, id_arr)

    colors = numpy_support.numpy_to_vtk(np.ascontiguousarray(all_rgb), deep=True)
    colors.SetName("Colors")

    pd = vtk.vtkPolyData()
    pd.SetPoints(points)
    pd.SetVerts(ca)
    pd.GetPointData().SetScalars(colors)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(pd)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(point_size)
    actor.GetProperty().SetRenderPointsAsSpheres(True)
    renderer.AddActor(actor)

    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.SetSize(size, size)
    render_window.AddRenderer(renderer)

    _setup_isometric_camera(renderer, azimuth=30, elevation=20)
    render_window.Render()
    _write_png(render_window, out_path)
    render_window.Finalize()
