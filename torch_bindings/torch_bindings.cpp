#include <memory>

#include "triangulation_bindings.h"
#include "utils/cuda_array.h"

namespace radfoam {

struct TorchBuffer : public OpaqueBuffer {
    torch::Tensor tensor;

    explicit TorchBuffer(size_t bytes) {
        // allocate on CUDA device
        // int64 dtype for alignment
        size_t num_words = (bytes + sizeof(int64_t) - 1) / sizeof(int64_t);
        tensor = torch::empty({(int64_t)num_words},
                              torch::dtype(torch::kInt64).device(torch::kCUDA));
    }

    void *data() override { return tensor.data_ptr(); }
};

std::unique_ptr<OpaqueBuffer> allocate_buffer(size_t bytes) {
    return std::make_unique<TorchBuffer>(bytes);
}

} // namespace radfoam

namespace foamsurf_topology_bindings {

PYBIND11_MODULE(torch_bindings, module) {
    module.doc() = "FoamSurf CUDA Delaunay topology bindings";

    init_triangulation_bindings(module);
}

} // namespace foamsurf_topology_bindings
