"""Generate labeled_marching_cubes.cpp from VTK triangle cases."""
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
VTK_CASES = HERE / "vtkMarchingCubesTriangleCases.cxx.txt"
text = VTK_CASES.read_text(encoding="utf-8")
rows = re.findall(r"\{\s*\{\s*((?:-?\d+\s*,\s*){15}-?\d+)\s*\}\s*\}", text)
assert len(rows) == 256, len(rows)
cases = []
for row in rows:
    nums = [int(x.strip()) for x in row.split(",")]
    assert len(nums) == 16
    cases.append(nums)

table_lines = ["static const int TRI_CASES[256][16] = {"]
for i, c in enumerate(cases):
    inner = ", ".join(f"{n:2d}" for n in c)
    comma = "," if i < 255 else ""
    table_lines.append(f"  {{{inner}}}{comma}")
table_lines.append("};")
table = "\n".join(table_lines)

cpp = r'''// Labeled Marching Cubes: VTK algorithm with custom vertex classification.
// Triangle cases / cube layout / interpolation follow vtkMarchingCubes
// (Kitware / Lorensen, BSD-3-Clause).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

''' + table + r'''

static const int CORNER[8][3] = {
    {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
    {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
};

// VTK edges: {0,1},{1,2},{3,2},{0,3},{4,5},{5,6},{7,6},{4,7},{0,4},{1,5},{3,7},{2,6}
static const int EDGES[12][2] = {
    {0, 1}, {1, 2}, {3, 2}, {0, 3}, {4, 5}, {5, 6},
    {7, 6}, {4, 7}, {0, 4}, {1, 5}, {3, 7}, {2, 6},
};

py::tuple marching_cubes_labeled(
    py::array_t<double, py::array::c_style | py::array::forcecast> volume,
    double isolevel,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> labels) {
    auto v = volume.unchecked<3>();
    auto lab = labels.unchecked<3>();
    const int nx = (int)v.shape(0);
    const int ny = (int)v.shape(1);
    const int nz = (int)v.shape(2);
    if (lab.shape(0) != nx || lab.shape(1) != ny || lab.shape(2) != nz) {
        throw std::invalid_argument("classified shape must match volume");
    }
    if (nx < 2 || ny < 2 || nz < 2) {
        throw std::invalid_argument("volume must be at least 2x2x2");
    }

    const size_t n_xedge = (size_t)(nx - 1) * (size_t)ny * (size_t)nz;
    const size_t n_yedge = (size_t)nx * (size_t)(ny - 1) * (size_t)nz;
    const size_t n_zedge = (size_t)nx * (size_t)ny * (size_t)(nz - 1);
    std::vector<int> x_edge(n_xedge, -1);
    std::vector<int> y_edge(n_yedge, -1);
    std::vector<int> z_edge(n_zedge, -1);
    const int x_stride_j = nx - 1;
    const int x_stride_k = (nx - 1) * ny;
    const int y_stride_j = nx;
    const int y_stride_k = nx * (ny - 1);
    const int z_stride_j = nx;
    const int z_stride_k = nx * ny;

    const size_t n_vox = (size_t)nx * (size_t)ny * (size_t)nz;
    size_t estimated = (size_t)std::pow((double)n_vox, 0.75);
    estimated = estimated < 1024 ? 1024 : estimated;

    std::vector<double> points;
    std::vector<int> triangles;
    points.reserve(estimated * 3);
    triangles.reserve(estimated * 3);

    const double value = isolevel;

    for (int k = 0; k < nz - 1; ++k) {
        for (int j = 0; j < ny - 1; ++j) {
            for (int i = 0; i < nx - 1; ++i) {
                int index = 0;
                int ii[8], jj[8], kk[8];
                for (int c = 0; c < 8; ++c) {
                    ii[c] = i + CORNER[c][0];
                    jj[c] = j + CORNER[c][1];
                    kk[c] = k + CORNER[c][2];
                    if (lab(ii[c], jj[c], kk[c])) {
                        index |= (1 << c);
                    }
                }
                if (index == 0 || index == 255) {
                    continue;
                }

                double s[8];
                double pts[8][3];
                for (int c = 0; c < 8; ++c) {
                    s[c] = v(ii[c], jj[c], kk[c]);
                    pts[c][0] = (double)ii[c];
                    pts[c][1] = (double)jj[c];
                    pts[c][2] = (double)kk[c];
                }

                const int* tri = TRI_CASES[index];
                for (int e = 0; e < 16 && tri[e] != -1; e += 3) {
                    int ids[3];
                    for (int t = 0; t < 3; ++t) {
                        const int e0 = EDGES[tri[e + t]][0];
                        const int e1 = EDGES[tri[e + t]][1];
                        const int ax = ii[e0];
                        const int ay = jj[e0];
                        const int az = kk[e0];
                        const int bx = ii[e1];
                        const int by = jj[e1];
                        const int bz = kk[e1];

                        int* slot;
                        if (ax != bx) {
                            const int imin = ax < bx ? ax : bx;
                            slot = &x_edge[(size_t)imin + (size_t)ay * x_stride_j +
                                           (size_t)az * x_stride_k];
                        } else if (ay != by) {
                            const int jmin = ay < by ? ay : by;
                            slot = &y_edge[(size_t)ax + (size_t)jmin * y_stride_j +
                                           (size_t)az * y_stride_k];
                        } else {
                            const int kmin = az < bz ? az : bz;
                            slot = &z_edge[(size_t)ax + (size_t)ay * z_stride_j +
                                           (size_t)kmin * z_stride_k];
                        }

                        if (*slot < 0) {
                            const double s0 = s[e0];
                            const double s1 = s[e1];
                            double tpar;
                            const double denom = s1 - s0;
                            if (denom == 0.0) {
                                tpar = 0.5;
                            } else {
                                tpar = (value - s0) / denom;
                            }
                            const int pid = (int)(points.size() / 3);
                            points.push_back(pts[e0][0] + tpar * (pts[e1][0] - pts[e0][0]));
                            points.push_back(pts[e0][1] + tpar * (pts[e1][1] - pts[e0][1]));
                            points.push_back(pts[e0][2] + tpar * (pts[e1][2] - pts[e0][2]));
                            *slot = pid;
                            ids[t] = pid;
                        } else {
                            ids[t] = *slot;
                        }
                    }
                    if (ids[0] != ids[1] && ids[0] != ids[2] && ids[1] != ids[2]) {
                        triangles.push_back(ids[0]);
                        triangles.push_back(ids[1]);
                        triangles.push_back(ids[2]);
                    }
                }
            }
        }
    }

    const py::ssize_t npts = (py::ssize_t)(points.size() / 3);
    const py::ssize_t ntri = (py::ssize_t)(triangles.size() / 3);
    py::array_t<double> pts_out({npts, (py::ssize_t)3});
    py::array_t<int> tri_out({ntri, (py::ssize_t)3});
    if (npts > 0) {
        std::memcpy(pts_out.mutable_data(), points.data(), points.size() * sizeof(double));
    }
    if (ntri > 0) {
        std::memcpy(tri_out.mutable_data(), triangles.data(), triangles.size() * sizeof(int));
    }
    return py::make_tuple(pts_out, tri_out);
}

PYBIND11_MODULE(labeled_marching_cubes, m) {
    m.doc() = "VTK Marching Cubes with custom vertex classification (C++)";
    m.def(
        "marching_cubes",
        &marching_cubes_labeled,
        py::arg("volume"),
        py::arg("isolevel"),
        py::arg("classified"),
        "Extract isosurface. volume: float64 (nx,ny,nz), classified: uint8 0/1 same shape.");
}
'''

out = HERE / "labeled_marching_cubes.cpp"
out.write_text(cpp, encoding="utf-8")
print("wrote", out, "bytes", out.stat().st_size)
