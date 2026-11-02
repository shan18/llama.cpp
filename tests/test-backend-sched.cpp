#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <vector>

// An in-place result must stay on its allocation's backend even when the
// immediately preceding operation runs on another backend.
static bool test_inplace_view(ggml_backend_t * backends, bool parallel) {
    auto * ctx0 = ggml_init({1024*1024, nullptr, true});
    auto * ctx1 = ggml_init({1024*1024, nullptr, true});
    auto * ctx  = ggml_init({1024*1024, nullptr, true});

    auto * w0 = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 8);
    auto * w1 = ggml_new_tensor_1d(ctx1, GGML_TYPE_F32, 4);
    auto buf0 = ggml_backend_alloc_ctx_tensors(ctx0, backends[0]);
    auto buf1 = ggml_backend_alloc_ctx_tensors(ctx1, backends[1]);
    GGML_ASSERT(buf0 && buf1);
    ggml_backend_buffer_set_usage(buf0, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(buf1, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    const float x[] = {1, 2, 3, 4, 5, 6, 7, 8};
    const float y[] = {100, 200, 300, 400};
    ggml_backend_tensor_set(w0, x, 0, sizeof(x));
    ggml_backend_tensor_set(w1, y, 0, sizeof(y));

    auto * base  = ggml_scale(ctx, w0, 2.0f);
    auto * chunk = ggml_scale(ctx, w1, 2.0f);
    auto * set   = ggml_set_inplace(ctx, base, chunk, base->nb[1], base->nb[2], base->nb[3], 2*sizeof(float));
    auto * view  = ggml_view_1d(ctx, set, 4, 2*sizeof(float));
    auto * out   = ggml_cont(ctx, view);
    ggml_set_name(base,  "base");
    ggml_set_name(chunk, "chunk");
    ggml_set_name(set,   "inplace");
    ggml_set_name(view,  "slice");
    ggml_set_name(out,   "result");
    ggml_set_output(out);

    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    auto sched = ggml_backend_sched_new(backends, nullptr, 3, 1024, parallel, true);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
    GGML_ASSERT(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);

    float actual[4];
    ggml_backend_tensor_get(out, actual, 0, sizeof(actual));
    bool ok = true;
    for (int i = 0; i < 4; ++i) {
        if (actual[i] != 2*y[i]) {
            std::fprintf(stderr, "parallel=%d index=%d: got %g, expected %g\n", parallel, i, actual[i], 2*y[i]);
            ok = false;
        }
    }

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_buffer_free(buf0);
    ggml_backend_buffer_free(buf1);
    ggml_free(ctx0);
    ggml_free(ctx1);
    return ok;
}

int main() {
    ggml_backend_load_all();
    std::vector<ggml_backend_dev_t> devices;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            devices.push_back(dev);
        }
    }
    if (devices.size() < 2) {
        std::printf("SKIP: scheduler in-place regression requires two GPU devices\n");
        return 77;
    }
    ggml_backend_t backends[] = {
        ggml_backend_dev_init(devices[0], nullptr),
        ggml_backend_dev_init(devices[1], nullptr),
        ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr),
    };
    for (auto backend : backends) {
        GGML_ASSERT(backend);
    }
    const bool serial_ok = test_inplace_view(backends, false);
    const bool parallel_ok = test_inplace_view(backends, true);
    for (auto backend : backends) {
        ggml_backend_free(backend);
    }
    return serial_ok && parallel_ok ? 0 : 1;
}
