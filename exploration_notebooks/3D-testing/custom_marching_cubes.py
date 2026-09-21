"""Marching Cubes with custom vertex classification.

The extract is a C++ port of vtkMarchingCubes: same cube layout, triangle
table, and linear interpolation. The cube case comes from your 0/1 labels,
not VTK's `scalar >= isolevel` test. When the labels match that test, the
mesh matches stock vtkMarchingCubes (same point/triangle counts; coordinates
agree to float32 precision, which is how VTK stores points).

Build the extension once (MSVC x64 Developer Prompt):

    python setup_labeled_mc.py build_ext --inplace
"""
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

try:
    from labeled_marching_cubes import marching_cubes as _mc_cpp
except ImportError as exc:
    raise ImportError(
        "labeled_marching_cubes C++ extension is not built. From "
        "exploration_notebooks/3D-testing, with the MSVC x64 environment "
        "loaded, run: python setup_labeled_mc.py build_ext --inplace"
    ) from exc


def classify_vertex_classical(input_val, isolevel):
    """VTK-style inside test: 1 iff value >= isolevel.

    Accepts a scalar or an array. Arrays stay vectorized; do not loop in Python.
    """
    inside = np.asarray(input_val) >= isolevel
    if inside.ndim == 0:
        return int(inside)
    return inside.astype(np.uint8, copy=False)


def classify_volume_classical(volume, isolevel):
    return (np.asarray(volume) >= isolevel).astype(np.uint8)


def _resolve_labels(volume, isolevel, classified, classify_fn):
    if classified is not None:
        return np.asarray(classified, dtype=np.uint8)
    if classify_fn is None:
        classify_fn = classify_vertex_classical
    try:
        labels = np.asarray(classify_fn(volume, isolevel))
        if labels.shape == volume.shape:
            return labels.astype(np.uint8, copy=False)
    except TypeError:
        pass
    labels = np.vectorize(lambda v: classify_fn(v, isolevel))(volume)
    return np.asarray(labels, dtype=np.uint8)


def numpy_volume_to_vtk_image(volume):
    volume = np.asarray(volume, dtype=np.float64)
    if volume.ndim != 3:
        raise ValueError("volume must be 3D")
    nx, ny, nz = volume.shape
    image = vtk.vtkImageData()
    image.SetDimensions(int(nx), int(ny), int(nz))
    image.SetSpacing(1.0, 1.0, 1.0)
    image.SetOrigin(0.0, 0.0, 0.0)
    scalars = numpy_to_vtk(
        np.ascontiguousarray(volume.ravel(order="F")),
        deep=True,
        array_type=vtk.VTK_DOUBLE,
    )
    scalars.SetName("scalars")
    image.GetPointData().SetScalars(scalars)
    return image


def vtk_marching_cubes(volume, isolevel, compute_normals=True):
    """Stock vtkMarchingCubes (VTK classifies vertices itself)."""
    image = numpy_volume_to_vtk_image(volume)
    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(image)
    mc.SetNumberOfContours(1)
    mc.SetValue(0, float(isolevel))
    if compute_normals:
        mc.ComputeNormalsOn()
    else:
        mc.ComputeNormalsOff()
    mc.ComputeGradientsOff()
    mc.ComputeScalarsOff()
    mc.Update()
    return mc.GetOutput()


def _numpy_mesh_to_vtk(points, triangles):
    poly = vtk.vtkPolyData()
    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetDataTypeToDouble()
    if len(points):
        vtk_pts.SetData(numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float64), deep=True))
    poly.SetPoints(vtk_pts)

    cells = vtk.vtkCellArray()
    if len(triangles):
        tris = np.ascontiguousarray(triangles, dtype=np.int64)
        n = tris.shape[0]
        connectivity = np.empty(n * 4, dtype=np.int64)
        connectivity[0::4] = 3
        connectivity[1::4] = tris[:, 0]
        connectivity[2::4] = tris[:, 1]
        connectivity[3::4] = tris[:, 2]
        cells.SetCells(n, numpy_to_vtkIdTypeArray(connectivity, deep=True))
    poly.SetPolys(cells)
    return poly


def marching_cubes(volume, isolevel, classified=None, classify_fn=None):
    """Extract an isosurface using custom vertex classification.

    volume: scalar field (nx, ny, nz) for edge interpolation.
    isolevel: contour value used in interpolation.
    classified: optional 0/1 labels, same shape as volume.
    classify_fn: used only if classified is None.

    Returns vtkPolyData from the C++ VTK-algorithm extract.
    """
    volume = np.ascontiguousarray(volume, dtype=np.float64)
    if volume.ndim != 3:
        raise ValueError("volume must be 3D")
    labels = _resolve_labels(volume, isolevel, classified, classify_fn)
    labels = np.ascontiguousarray(labels, dtype=np.uint8)
    if labels.shape != volume.shape:
        raise ValueError("classified shape must match volume")
    points, triangles = _mc_cpp(volume, float(isolevel), labels)
    return _numpy_mesh_to_vtk(points, triangles)
