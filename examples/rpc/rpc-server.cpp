#include "ggml-cpu.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#ifdef GGML_USE_SYCL
#include "ggml-sycl.h"
#endif

#include "ggml-rpc.h"
#ifdef _WIN32
#  include <windows.h>
#else
#  include <unistd.h>
#endif
#include <string>
#include <stdio.h>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

const int MAX_GGUF_PATH = 16;

struct rpc_server_params {
    std::string              host        = "127.0.0.1";
    int                      port        = 50052;
    size_t                   backend_mem = 0;
    std::string              cache_dir   = "";
    std::vector<std::string> gguf_path;
};

static void print_usage(int /*argc*/, char ** argv, rpc_server_params params) {
    fprintf(stderr, "Usage: %s [options]\n\n", argv[0]);
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help               show this help message and exit\n");
    fprintf(stderr, "  -H HOST, --host HOST     host to bind to (default: %s)\n", params.host.c_str());
    fprintf(stderr, "  -p PORT, --port PORT     port to bind to (default: %d)\n", params.port);
    fprintf(stderr, "  -f PATH, --gguf PATH     path to GGUF file\n");
    fprintf(stderr, "  -d DIR, --cache-dir DIR  local cache dir\n");
    fprintf(stderr, "  -m MEM, --mem MEM        backend memory size (in MB)\n");
    fprintf(stderr, "\n");
}

static bool rpc_server_params_parse(int argc, char ** argv, rpc_server_params & params) {
    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg == "-H" || arg == "--host") {
            if (++i >= argc) {
                return false;
            }
            params.host = argv[i];
        } else if (arg == "-p" || arg == "--port") {
            if (++i >= argc) {
                return false;
            }
            params.port = std::stoi(argv[i]);
            if (params.port <= 0 || params.port > 65535) {
                return false;
            }
        } else if (arg == "-f" || arg == "--gguf") {
            if (++i >= argc) {
                return false;
            }
            if (params.gguf_path.size() >= MAX_GGUF_PATH) {
                fprintf(stderr, "error: too many GGUF files\n");
                return false;
            }
            fs::path gguf_path(argv[i]);
            if (!fs::is_regular_file(gguf_path)) {
                fprintf(stderr, "error: GGUF file does not exist: %s\n", gguf_path.c_str());
                return false;
            }
            params.gguf_path.push_back(argv[i]);
        } else if (arg == "-d" || arg == "--cache-dir") {
            if (++i >= argc) {
                return false;
            }
            fs::path cache_dir(argv[i]);
            if (!fs::is_directory(cache_dir)) {
                fprintf(stderr, "error: cache dir does not exist: %s\n", cache_dir.c_str());
                return false;
            }
            params.cache_dir = argv[i];
        } else if (arg == "-m" || arg == "--mem") {
            if (++i >= argc) {
                return false;
            }
            params.backend_mem = std::stoul(argv[i]) * 1024 * 1024;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv, params);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argc, argv, params);
            exit(0);
        }
    }
    return true;
}

static ggml_backend_t create_backend() {
    ggml_backend_t backend = NULL;
#ifdef GGML_USE_CUDA
    fprintf(stderr, "%s: using CUDA backend\n", __func__);
    backend = ggml_backend_cuda_init(0); // init device 0
    if (!backend) {
        fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
    }
#elif GGML_USE_METAL
    fprintf(stderr, "%s: using Metal backend\n", __func__);
    backend = ggml_backend_metal_init();
    if (!backend) {
        fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
    }
#elif GGML_USE_VULKAN
    fprintf(stderr, "%s: using Vulkan backend\n", __func__);
    backend = ggml_backend_vk_init(0); // init device 0
    if (!backend) {
        fprintf(stderr, "%s: ggml_backend_vulkan_init() failed\n", __func__);
    }
#elif GGML_USE_SYCL
    fprintf(stderr, "%s: using SYCL backend\n", __func__);
    backend = ggml_backend_sycl_init(0); // init device 0
    if (!backend) {
        fprintf(stderr, "%s: ggml_backend_sycl_init() failed\n", __func__);
    }
#endif

    // if there aren't GPU Backends fallback to CPU backend
    if (!backend) {
        fprintf(stderr, "%s: using CPU backend\n", __func__);
        backend = ggml_backend_cpu_init();
    }
    return backend;
}

static void get_backend_memory(size_t * free_mem, size_t * total_mem) {
#ifdef GGML_USE_CUDA
    ggml_backend_cuda_get_device_memory(0, free_mem, total_mem);
#elif GGML_USE_VULKAN
    ggml_backend_vk_get_device_memory(0, free_mem, total_mem);
#elif GGML_USE_SYCL
    ggml_backend_sycl_get_device_memory(0, free_mem, total_mem);
#else
    #ifdef _WIN32
        MEMORYSTATUSEX status;
        status.dwLength = sizeof(status);
        GlobalMemoryStatusEx(&status);
        *total_mem = status.ullTotalPhys;
        *free_mem = status.ullAvailPhys;
    #else
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        *total_mem = pages * page_size;
        *free_mem = *total_mem;
    #endif
#endif
}

int main(int argc, char * argv[]) {
    rpc_server_params params;
    if (!rpc_server_params_parse(argc, argv, params)) {
        fprintf(stderr, "Invalid parameters\n");
        return 1;
    }

    if (params.host != "127.0.0.1") {
        fprintf(stderr, "\n");
        fprintf(stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        fprintf(stderr, "WARNING: Host ('%s') is != '127.0.0.1'\n", params.host.c_str());
        fprintf(stderr, "         Never expose the RPC server to an open network!\n");
        fprintf(stderr, "         This is an experimental feature and is not secure!\n");
        fprintf(stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        fprintf(stderr, "\n");
    }

    ggml_backend_t backend = create_backend();
    if (!backend) {
        fprintf(stderr, "Failed to create backend\n");
        return 1;
    }
    std::string endpoint = params.host + ":" + std::to_string(params.port);
    size_t free_mem, total_mem;
    if (params.backend_mem > 0) {
        free_mem = params.backend_mem;
        total_mem = params.backend_mem;
    } else {
        get_backend_memory(&free_mem, &total_mem);
    }
    const char * gguf_path[MAX_GGUF_PATH] = {0};
    int n_gguf_path = params.gguf_path.size();
    for (size_t i = 0; i < n_gguf_path; i++) {
        gguf_path[i] = params.gguf_path[i].c_str();
    }
    const char * cache_dir = params.cache_dir.empty() ? nullptr : params.cache_dir.c_str();
    printf("Starting RPC server on %s, backend memory: %zu MB\n", endpoint.c_str(), free_mem / (1024 * 1024));
    ggml_backend_rpc_start_server(backend, endpoint.c_str(), cache_dir, n_gguf_path, gguf_path, free_mem, total_mem);
    ggml_backend_free(backend);
    return 0;
}
