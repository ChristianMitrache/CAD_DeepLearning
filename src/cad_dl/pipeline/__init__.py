"""Dataset-agnostic CAD preprocessing pipeline.

Every dataset (AutoMate, ABC, Fusion360, ...) is standardized into this
on-disk format:

    <processed_root>/<dataset>/<id>/scene.ply      canonical merged mesh
    <processed_root>/<dataset>/<id>/points.npz     cached point cloud
    <processed_root>/<dataset>/<id>/metadata.json  see cad_dl.pipeline.schema
    <processed_root>/<dataset>/index.parquet       one row per assembly

Never touch these files directly — use `write_assembly`, `load_metadata`,
`load_scene_mesh`, `load_points`, or `validate_assembly` from `cad_dl.pipeline.io`.

To add a new dataset: subclass `cad_dl.pipeline.dataset.Dataset`, implement
`download` + `iter_ids` + `load_scene`, decorate with `@register(name)`.
"""
from cad_dl.pipeline.dataset import DATASETS, Dataset, get_dataset, register
from cad_dl.pipeline.io import (
    SampledPoints,
    load_metadata,
    load_points,
    load_scene_mesh,
    validate_assembly,
    write_assembly,
)
from cad_dl.pipeline.sampling import resample_from_disk, sample_scene
from cad_dl.pipeline.schema import SCHEMA_VERSION, AssemblyMetadata, PartRecord

__all__ = [
    "DATASETS",
    "SCHEMA_VERSION",
    "AssemblyMetadata",
    "Dataset",
    "PartRecord",
    "SampledPoints",
    "get_dataset",
    "load_metadata",
    "load_points",
    "load_scene_mesh",
    "register",
    "resample_from_disk",
    "sample_scene",
    "validate_assembly",
    "write_assembly",
]
