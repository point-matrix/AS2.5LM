#include <torch/extension.h>

#include <vector>

std::vector<at::Tensor> grid_engine_run(at::Tensor xyz, at::Tensor labels, at::Tensor conf,
                                        at::Tensor pri_lut, at::Tensor trav_lut,
                                        std::vector<double> params, bool profile,
                                        at::Tensor pack_layout, int64_t pack_itemsize);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &grid_engine_run,
        "Grid Engine v2 on CUDA. Returns [matrix (n,20) f64, xy_bounds (4), stats (7), stage_ms (5), "
        "packed FINAL_DTYPE bytes]",
        py::arg("xyz"), py::arg("labels"), py::arg("conf"), py::arg("pri_lut"),
        py::arg("trav_lut"), py::arg("params"), py::arg("profile"),
        py::arg("pack_layout"), py::arg("pack_itemsize"));
}
