#include <torch/extension.h>

#include "qff.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qff_forward", &qff_forward, "qff forward (CUDA)");
    m.def("qff_backward", &qff_backward, "qff backward (CUDA)");
    m.def("qff_backward_backward", &qff_backward_backward, "qff backward backward (CUDA)");
}