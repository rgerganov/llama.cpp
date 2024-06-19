#include "ggml-rpc.h"
#include "ggml.h"
#include "ggml-backend-impl.h"

#include <cinttypes>
#include <string>
#include <vector>
#include <queue>
#include <memory>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <unordered_map>
#include <unordered_set>
#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#     define NOMINMAX
#  endif
#  include <windows.h>
#  include <winsock2.h>
#else
#  include <signal.h>
#  include <arpa/inet.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <netdb.h>
#  include <unistd.h>
#endif
#include <string.h>

#define UNUSED GGML_UNUSED

#define GGML_DEBUG 0
#if (GGML_DEBUG >= 1)
#define GGML_PRINT_DEBUG(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG(...)
#endif

#ifdef _WIN32
typedef SOCKET sockfd_t;
using ssize_t = __int64;
#else
typedef int sockfd_t;
#endif

// cross-platform socket
struct socket_t {
    sockfd_t fd;
    socket_t(sockfd_t fd) : fd(fd) {}
    ~socket_t() {
        GGML_PRINT_DEBUG("[%s] closing socket %d\n", __func__, this->fd);
#ifdef _WIN32
        closesocket(this->fd);
#else
        close(this->fd);
#endif
    }
};

// ggml_tensor is serialized into rpc_tensor
#pragma pack(push, 1)
struct rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    char name[GGML_MAX_NAME];

    char padding[4];
};
#pragma pack(pop)

static_assert(sizeof(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
enum rpc_cmd {
    HELLO = 0,
    ALLOC_BUFFER,
    GET_ALIGNMENT,
    GET_MAX_SIZE,
    BUFFER_GET_BASE,
    FREE_BUFFER,
    BUFFER_CLEAR,
    SET_TENSOR,
    GET_TENSOR,
    COPY_TENSOR,
    REMOTE_COPY_TENSOR,
    GRAPH_COMPUTE,
    GET_DEVICE_MEMORY,
    FREE_ALL_BUFFERS,
};

enum rpc_actor {
    CLIENT = 0,
    SERVER,
};

// RPC data structures

static ggml_guid_t ggml_backend_rpc_guid() {
    static ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    return &guid;
}

struct ggml_backend_rpc_buffer_type_context {
    std::string endpoint;
    std::string name;
    size_t alignment;
    size_t max_size;
};

struct ggml_backend_rpc_context {
    std::string endpoint;
    std::string name;
};

struct ggml_backend_rpc_buffer_context {
    std::string endpoint;
    std::shared_ptr<socket_t> sock;
    std::unordered_map<ggml_backend_buffer_t, void *> base_cache;
    uint64_t remote_ptr;
    std::string name;
};

// RPC helper functions

static std::shared_ptr<socket_t> make_socket(sockfd_t fd) {
#ifdef _WIN32
    if (fd == INVALID_SOCKET) {
        return nullptr;
    }
#else
    if (fd < 0) {
        return nullptr;
    }
#endif
    return std::make_shared<socket_t>(fd);
}

static bool set_no_delay(sockfd_t sockfd) {
    int flag = 1;
    // set TCP_NODELAY to disable Nagle's algorithm
    int ret = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (char *)&flag, sizeof(int));
    return ret == 0;
}

static bool set_reuse_addr(sockfd_t sockfd) {
    int flag = 1;
    int ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (char *)&flag, sizeof(int));
    return ret == 0;
}

static std::shared_ptr<socket_t> socket_connect(const char * host, int port) {
    struct sockaddr_in addr;
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    auto sock_ptr = make_socket(sockfd);
    if (sock_ptr == nullptr) {
        return nullptr;
    }
    if (!set_no_delay(sockfd)) {
        fprintf(stderr, "Failed to set TCP_NODELAY\n");
        return nullptr;
    }
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    struct hostent * server = gethostbyname(host);
    if (server == NULL) {
        fprintf(stderr, "Cannot resolve host '%s'\n", host);
        return nullptr;
    }
    memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);
    if (connect(sock_ptr->fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        return nullptr;
    }
    return sock_ptr;
}

static std::shared_ptr<socket_t> socket_accept(sockfd_t srv_sockfd) {
    auto client_socket_fd = accept(srv_sockfd, NULL, NULL);
    auto client_socket = make_socket(client_socket_fd);
    if (client_socket == nullptr) {
        return nullptr;
    }
    if (!set_no_delay(client_socket_fd)) {
        fprintf(stderr, "Failed to set TCP_NODELAY\n");
        return nullptr;
    }
    return client_socket;
}

static std::shared_ptr<socket_t> create_server_socket(const char * host, int port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    auto sock = make_socket(sockfd);
    if (sock == nullptr) {
        return nullptr;
    }
    if (!set_reuse_addr(sockfd)) {
        fprintf(stderr, "Failed to set SO_REUSEADDR\n");
        return nullptr;
    }
    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_port = htons(port);

    if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
        return nullptr;
    }
    if (listen(sockfd, 2) < 0) {
        return nullptr;
    }
    return sock;
}

static bool send_data(sockfd_t sockfd, const void * data, size_t size) {
    size_t bytes_sent = 0;
    while (bytes_sent < size) {
        ssize_t n = send(sockfd, (const char *)data + bytes_sent, size - bytes_sent, 0);
        if (n < 0) {
            return false;
        }
        bytes_sent += n;
    }
    return true;
}

static bool recv_data(sockfd_t sockfd, void * data, size_t size) {
    size_t bytes_recv = 0;
    while (bytes_recv < size) {
        ssize_t n = recv(sockfd, (char *)data + bytes_recv, size - bytes_recv, 0);
        if (n <= 0) {
            return false;
        }
        bytes_recv += n;
    }
    return true;
}

static bool parse_endpoint(const std::string & endpoint, std::string & host, int & port) {
    size_t pos = endpoint.find(':');
    if (pos == std::string::npos) {
        return false;
    }
    host = endpoint.substr(0, pos);
    port = std::stoi(endpoint.substr(pos + 1));
    return true;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
static bool send_rpc_cmd(const std::shared_ptr<socket_t> & sock, enum rpc_cmd cmd, const std::vector<uint8_t> & input, std::vector<uint8_t> & output) {
    uint8_t cmd_byte = cmd;
    if (!send_data(sock->fd, &cmd_byte, sizeof(cmd_byte))) {
        return false;
    }
    uint64_t input_size = input.size();
    if (!send_data(sock->fd, &input_size, sizeof(input_size))) {
        return false;
    }
    if (!send_data(sock->fd, input.data(), input.size())) {
        return false;
    }
    uint64_t output_size;
    if (!recv_data(sock->fd, &output_size, sizeof(output_size))) {
        return false;
    }
    if (output_size == 0) {
        output.clear();
        return true;
    }
    output.resize(output_size);
    if (!recv_data(sock->fd, output.data(), output_size)) {
        return false;
    }
    return true;
}

// RPC client-side implementation

static void send_hello(std::shared_ptr<socket_t> sock, rpc_actor actor) {
    // input serialization format: | actor (1 byte) |
    std::vector<uint8_t> input(1, actor);
    std::vector<uint8_t> output;
    bool status = send_rpc_cmd(sock, HELLO, input, output);
    GGML_ASSERT(status);
    GGML_ASSERT(output.empty());
}

static std::shared_ptr<socket_t> get_socket(const std::string & endpoint) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static std::unordered_map<std::string, std::weak_ptr<socket_t>> sockets;
    static bool initialized = false;

    auto it = sockets.find(endpoint);
    if (it != sockets.end()) {
        if (auto sock = it->second.lock()) {
            return sock;
        }
    }
    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        return nullptr;
    }
#ifdef _WIN32
    if (!initialized) {
        WSADATA wsaData;
        int res = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if (res != 0) {
            return nullptr;
        }
        initialized = true;
    }
#else
    UNUSED(initialized);
#endif
    auto sock = socket_connect(host.c_str(), port);
    if (sock == nullptr) {
        return nullptr;
    }
    send_hello(sock, CLIENT);
    GGML_PRINT_DEBUG("[%s] connected to %s, sockfd=%d\n", __func__, endpoint.c_str(), sock->fd);
    sockets[endpoint] = sock;
    return sock;
}

GGML_CALL static const char * ggml_backend_rpc_buffer_get_name(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    return ctx->name.c_str();
}

GGML_CALL static void ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    // input serialization format: | remote_ptr (8 bytes) |
    std::vector<uint8_t> input(sizeof(uint64_t), 0);
    uint64_t remote_ptr = ctx->remote_ptr;
    memcpy(input.data(), &remote_ptr, sizeof(remote_ptr));
    std::vector<uint8_t> output;
    bool status = send_rpc_cmd(ctx->sock, FREE_BUFFER, input, output);
    GGML_ASSERT(status);
    GGML_ASSERT(output.empty());
    delete ctx;
}

GGML_CALL static void * ggml_backend_rpc_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    if (ctx->base_cache.find(buffer) != ctx->base_cache.end()) {
        return ctx->base_cache[buffer];
    }
    // input serialization format: | remote_ptr (8 bytes) |
    std::vector<uint8_t> input(sizeof(uint64_t), 0);
    uint64_t remote_ptr = ctx->remote_ptr;
    memcpy(input.data(), &remote_ptr, sizeof(remote_ptr));
    std::vector<uint8_t> output;
    bool status = send_rpc_cmd(ctx->sock, BUFFER_GET_BASE, input, output);
    GGML_ASSERT(status);
    GGML_ASSERT(output.size() == sizeof(uint64_t));
    // output serialization format: | base_ptr (8 bytes) |
    uint64_t base_ptr;
    memcpy(&base_ptr, output.data(), sizeof(base_ptr));
    void * base = reinterpret_cast<void *>(base_ptr);
    ctx->base_cache[buffer] = base;
    return base;
}

static rpc_tensor serialize_tensor(const ggml_tensor * tensor) {
    rpc_tensor result;
    result.id = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    if (tensor->buffer) {
        ggml_backend_buffer_t buffer = tensor->buffer;
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        result.buffer = ctx->remote_ptr;
    } else {
        result.buffer = 0;
    }
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    for (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;
    result.data = reinterpret_cast<uint64_t>(tensor->data);
    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    return result;
}

GGML_CALL static void ggml_backend_rpc_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    UNUSED(buffer);
    if (ggml_is_quantized(tensor->type)) {
        // TODO: this check is due to MATRIX_ROW_PADDING in CUDA and should be generalized
        GGML_ASSERT(tensor->ne[0] % 512 == 0 && "unsupported quantized tensor");
    }
}

GGML_CALL static void ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    // input serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes) |
    size_t input_size = sizeof(rpc_tensor) + sizeof(uint64_t) + size;
    std::vector<uint8_t> input(input_size, 0);
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    memcpy(input.data(), &rpc_tensor, sizeof(rpc_tensor));
    memcpy(input.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
    memcpy(input.data() + sizeof(rpc_tensor) + sizeof(offset), data, size);
    std::vector<uint8_t> output;
    bool status = send_rpc_cmd(ctx->sock, SET_TENSOR, input, output);
    GGML_ASSERT(status);
}

GGML_CALL static void ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    // input serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) |
    int input_size = sizeof(rpc_tensor) + 2*sizeof(uint64_t);
    std::vector<uint8_t> input(input_size, 0);
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    memcpy(input.data(), &rpc_tensor, sizeof(rpc_tensor));
    memcpy(input.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
    memcpy(input.data() + sizeof(rpc_tensor) + sizeof(offset), &size, sizeof(size));
    std::vector<uint8_t> output;
    bool status = send_rpc_cmd(ctx->sock, GET_TENSOR, input, output);
    GGML_ASSERT(status);
    GGML_ASSERT(output.size() == size);
    // output serialization format: | data (size bytes) |
    memcpy(data, output.data(), size);
}

static bool remote_copy_tensor(const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t src_buffer = src->buffer;
    ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *)src_buffer->context;
    ggml_backend_buffer_t dst_buffer = dst->buffer;
    ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *)dst_buffer->context;
    // input serialization format: | rpc_tensor src | rpc_tensor dst | dst_endpoint_size (4 bytes) | dst_endpoint (dst_endpoint_size bytes) |
    int input_size = 2*sizeof(rpc_tensor) + sizeof(uint32_t) + dst_ctx->endpoint.size();
    std::vector<uint8_t> input(input_size, 0);
    rpc_tensor rpc_src = serialize_tensor(src);
    rpc_tensor rpc_dst = serialize_tensor(dst);
    memcpy(input.data(), &rpc_src, sizeof(rpc_src));
    memcpy(input.data() + sizeof(rpc_src), &rpc_dst, sizeof(rpc_dst));
    uint32_t dst_endpoint_size = dst_ctx->endpoint.size();
    memcpy(input.data() + 2*sizeof(rpc_tensor), &dst_endpoint_size, sizeof(dst_endpoint_size));
    memcpy(input.data() + 2*sizeof(rpc_tensor) + sizeof(dst_endpoint_size), dst_ctx->endpoint.c_str(), dst_endpoint_size);
    std::vector<uint8_t> output;
    bool status = send_rpc_cmd(src_ctx->sock, REMOTE_COPY_TENSOR, input, output);
    GGML_ASSERT(status);
    // output serialization format: | result (1 byte) |
    GGML_ASSERT(output.size() == 1);
    return output[0];
}

GGML_CALL static bool ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    // check if src and dst are on the same server
    ggml_backend_buffer_t src_buffer = src->buffer;
    ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *)src_buffer->context;
    ggml_backend_buffer_t dst_buffer = dst->buffer;
    ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *)dst_buffer->context;
    if (src_ctx->sock != dst_ctx->sock) {
        return remote_copy_tensor(src, dst);
    }
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    // input serialization format: | rpc_tensor src | rpc_tensor dst |
    int input_size = 2*sizeof(rpc_tensor);
    std::vector<uint8_t> input(input_size, 0);
    rpc_tensor rpc_src = serialize_tensor(src);
    rpc_tensor rpc_dst = serialize_tensor(dst);
    memcpy(input.data(), &rpc_src, sizeof(rpc_src));
    memcpy(input.data() + sizeof(rpc_src), &rpc_dst, sizeof(rpc_dst));
    std::vector<uint8_t> output;
    bool status = send_rpc_cmd(ctx->sock, COPY_TENSOR, input, output);
    GGML_ASSERT(status);
    // output serialization format: | result (1 byte) |
    GGML_ASSERT(output.size() == 1);
    return output[0];
}

GGML_CALL static void ggml_backend_rpc_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    // serialization format: | bufptr (8 bytes) | value (1 byte) |
    int input_size = sizeof(uint64_t) + sizeof(uint8_t);
    std::vector<uint8_t> input(input_size, 0);
    memcpy(input.data(), &ctx->remote_ptr, sizeof(ctx->remote_ptr));
    memcpy(input.data() + sizeof(ctx->remote_ptr), &value, sizeof(value));
    std::vector<uint8_t> output;
    bool status = send_rpc_cmd(ctx->sock, BUFFER_CLEAR, input, output);
    GGML_ASSERT(status);
}

static ggml_backend_buffer_i ggml_backend_rpc_buffer_interface = {
    /* .get_name        = */ ggml_backend_rpc_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_rpc_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rpc_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rpc_buffer_init_tensor,
    /* .set_tensor      = */ ggml_backend_rpc_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_rpc_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_rpc_buffer_clear,
    /* .reset           = */ NULL,
};

GGML_CALL static const char * ggml_backend_rpc_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->name.c_str();
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_rpc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    // input serialization format: | size (8 bytes) |
    int input_size = sizeof(uint64_t);
    std::vector<uint8_t> input(input_size, 0);
    memcpy(input.data(), &size, sizeof(size));
    std::vector<uint8_t> output;
    auto sock = get_socket(buft_ctx->endpoint);
    bool status = send_rpc_cmd(sock, ALLOC_BUFFER, input, output);
    GGML_ASSERT(status);
    GGML_ASSERT(output.size() == 2*sizeof(uint64_t));
    // output serialization format: | remote_ptr (8 bytes) | remote_size (8 bytes) |
    uint64_t remote_ptr;
    memcpy(&remote_ptr, output.data(), sizeof(remote_ptr));
    size_t remote_size;
    memcpy(&remote_size, output.data() + sizeof(uint64_t), sizeof(remote_size));
    if (remote_ptr != 0) {
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
            ggml_backend_rpc_buffer_interface,
            new ggml_backend_rpc_buffer_context{buft_ctx->endpoint, sock, {}, remote_ptr, "RPC[" + std::string(buft_ctx->endpoint) + "]"},
            remote_size);
        return buffer;
    } else {
        return nullptr;
    }
}

static size_t get_alignment(const std::shared_ptr<socket_t> & sock) {
    // input serialization format: | 0 bytes |
    std::vector<uint8_t> input;
    std::vector<uint8_t> output;
    bool status = send_rpc_cmd(sock, GET_ALIGNMENT, input, output);
    GGML_ASSERT(status);
    GGML_ASSERT(output.size() == sizeof(uint64_t));
    // output serialization format: | alignment (8 bytes) |
    uint64_t alignment;
    memcpy(&alignment, output.data(), sizeof(alignment));
    return alignment;
}

GGML_CALL static size_t ggml_backend_rpc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->alignment;
}

static size_t get_max_size(const std::shared_ptr<socket_t> & sock) {
    // input serialization format: | 0 bytes |
    std::vector<uint8_t> input;
    std::vector<uint8_t> output;
    bool status = send_rpc_cmd(sock, GET_MAX_SIZE, input, output);
    GGML_ASSERT(status);
    GGML_ASSERT(output.size() == sizeof(uint64_t));
    // output serialization format: | max_size (8 bytes) |
    uint64_t max_size;
    memcpy(&max_size, output.data(), sizeof(max_size));
    return max_size;
}

GGML_CALL static size_t ggml_backend_rpc_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->max_size;
}

GGML_CALL static size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    UNUSED(buft);
    return ggml_nbytes(tensor);
}

static ggml_backend_buffer_type_i ggml_backend_rpc_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_rpc_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_rpc_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_rpc_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_rpc_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_rpc_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

GGML_CALL static const char * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    return rpc_ctx->name.c_str();
}

GGML_CALL static void ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    delete rpc_ctx;
    delete backend;
}

GGML_CALL static ggml_backend_buffer_type_t ggml_backend_rpc_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    return ggml_backend_rpc_buffer_type(ctx->endpoint.c_str());
}

GGML_CALL static void ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    UNUSED(backend);
    // this is no-op because we don't have any async operations
}

static void add_tensor(ggml_tensor * tensor, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
    if (tensor == nullptr) {
        return;
    }
    if (visited.find(tensor) != visited.end()) {
        return;
    }
    visited.insert(tensor);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        add_tensor(tensor->src[i], tensors, visited);
    }
    add_tensor(tensor->view_src, tensors, visited);
    tensors.push_back(serialize_tensor(tensor));
}

static void serialize_graph(const ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    for (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph->nodes[i], tensors, visited);
    }
    // serialization format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    int output_size = sizeof(uint32_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor);
    output.resize(output_size, 0);
    memcpy(output.data(), &n_nodes, sizeof(n_nodes));
    for (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(output.data() + sizeof(n_nodes) + i * sizeof(uint64_t), &cgraph->nodes[i], sizeof(uint64_t));
    }
    uint32_t * out_ntensors = (uint32_t *)(output.data() + sizeof(n_nodes) + n_nodes * sizeof(uint64_t));
    *out_ntensors = n_tensors;
    rpc_tensor * out_tensors = (rpc_tensor *)(output.data() + sizeof(n_nodes) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t));
    memcpy(out_tensors, tensors.data(), n_tensors * sizeof(rpc_tensor));
}

GGML_CALL static enum ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    std::vector<uint8_t> input;
    serialize_graph(cgraph, input);
    std::vector<uint8_t> output;
    auto sock = get_socket(rpc_ctx->endpoint);
    bool status = send_rpc_cmd(sock, GRAPH_COMPUTE, input, output);
    GGML_ASSERT(status);
    GGML_ASSERT(output.size() == 1);
    return (enum ggml_status)output[0];
}

GGML_CALL static bool ggml_backend_rpc_supports_op(ggml_backend_t backend, const ggml_tensor * op) {
    UNUSED(backend);
    UNUSED(op);
    //TODO: call the remote backend and cache the results
    return true;
}

GGML_CALL static bool ggml_backend_rpc_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    if (buft->iface.get_name != ggml_backend_rpc_buffer_type_name) {
        return false;
    }
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    return buft_ctx->endpoint == rpc_ctx->endpoint;
}

static ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .get_default_buffer_type = */ ggml_backend_rpc_get_default_buffer_type,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .supports_op             = */ ggml_backend_rpc_supports_op,
    /* .supports_buft           = */ ggml_backend_rpc_supports_buft,
    /* .offload_op              = */ NULL,
    /* .event_new               = */ NULL,
    /* .event_free              = */ NULL,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .event_synchronize       = */ NULL,
};

GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    // NOTE: buffer types are allocated and never freed; this is by design
    static std::unordered_map<std::string, ggml_backend_buffer_type_t> buft_map;
    auto it = buft_map.find(endpoint);
    if (it != buft_map.end()) {
        return it->second;
    }
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        return nullptr;
    }
    size_t alignment = get_alignment(sock);
    size_t max_size = get_max_size(sock);
    ggml_backend_rpc_buffer_type_context * buft_ctx = new ggml_backend_rpc_buffer_type_context {
        /* .endpoint  = */ endpoint,
        /* .name      = */ "RPC[" + std::string(endpoint) + "]",
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };

    ggml_backend_buffer_type_t buft = new ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_rpc_buffer_type_interface,
        /* .context = */ buft_ctx
    };
    buft_map[endpoint] = buft;
    return buft;
}

GGML_CALL ggml_backend_t ggml_backend_rpc_init(const char * endpoint) {
    ggml_backend_rpc_context * ctx = new ggml_backend_rpc_context {
        /* .endpoint  = */ endpoint,
        /* .name      = */ "RPC[" + std::string(endpoint) + "]",
    };

    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */ ggml_backend_rpc_guid(),
        /* .interface = */ ggml_backend_rpc_interface,
        /* .context   = */ ctx
    };
    return backend;
}

GGML_API GGML_CALL bool ggml_backend_is_rpc(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

static void get_device_memory(const std::shared_ptr<socket_t> & sock, size_t * free, size_t * total) {
    // input serialization format: | 0 bytes |
    std::vector<uint8_t> input;
    std::vector<uint8_t> output;
    bool status = send_rpc_cmd(sock, GET_DEVICE_MEMORY, input, output);
    GGML_ASSERT(status);
    GGML_ASSERT(output.size() == 2*sizeof(uint64_t));
    // output serialization format: | free (8 bytes) | total (8 bytes) |
    uint64_t free_mem;
    memcpy(&free_mem, output.data(), sizeof(free_mem));
    uint64_t total_mem;
    memcpy(&total_mem, output.data() + sizeof(uint64_t), sizeof(total_mem));
    *free = free_mem;
    *total = total_mem;
}

GGML_API GGML_CALL void ggml_backend_rpc_get_device_memory(const char * endpoint, size_t * free, size_t * total) {
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        *free = 0;
        *total = 0;
        return;
    }
    get_device_memory(sock, free, total);
}

// RPC server-side implementation

template <typename T>
class message_queue {
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cvar;

public:
    message_queue() {}

    void push(const T &value) {
        std::unique_lock<std::mutex> lock(mutex);
        queue.push(value);
        lock.unlock();
        cvar.notify_all();
    }

    void pop(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        cvar.wait(lock, [this] { return queue.size() > 0; });
        *out = queue.front();
        queue.pop();
    }
};

struct rpc_response {
    std::vector<uint8_t> output;
    bool status;
};

using rpc_response_ptr = std::shared_ptr<rpc_response>;
using response_queue = message_queue<rpc_response_ptr>;
using response_queue_ptr = std::shared_ptr<response_queue>;

struct rpc_request {
    rpc_cmd cmd;
    std::vector<uint8_t> input;
    response_queue_ptr response_queue;
};
using rpc_request_ptr = std::shared_ptr<rpc_request>;
using request_queue = message_queue<rpc_request_ptr>;
using request_queue_ptr = std::shared_ptr<request_queue>;

class rpc_server {
public:
    rpc_server(ggml_backend_t backend) : backend(backend) {}
    ~rpc_server();

    bool alloc_buffer(const std::vector<uint8_t> & input, std::vector<uint8_t> & output);
    void get_alignment(std::vector<uint8_t> & output);
    void get_max_size(std::vector<uint8_t> & output);
    bool buffer_get_base(const std::vector<uint8_t> & input, std::vector<uint8_t> & output);
    bool free_buffer(const std::vector<uint8_t> & input);
    bool buffer_clear(const std::vector<uint8_t> & input);
    bool set_tensor(const std::vector<uint8_t> & input);
    void remote_set_tensor(std::shared_ptr<socket_t> sock, const rpc_tensor * rpc_src, const rpc_tensor * rpc_dst);
    bool get_tensor(const std::vector<uint8_t> & input, std::vector<uint8_t> & output);
    bool copy_tensor(const std::vector<uint8_t> & input, std::vector<uint8_t> & output);
    bool remote_copy_tensor(const std::vector<uint8_t> & input, std::vector<uint8_t> & output);
    bool graph_compute(const std::vector<uint8_t> & input, std::vector<uint8_t> & output);

    void free_all_buffers();
private:
    ggml_tensor * deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor);
    ggml_tensor * create_node(uint64_t id,
                              struct ggml_context * ctx,
                              const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map);


    ggml_backend_t backend;
    std::unordered_set<ggml_backend_buffer_t> buffers;
};

bool rpc_server::alloc_buffer(const std::vector<uint8_t> & input, std::vector<uint8_t> & output) {
    // input serialization format: | size (8 bytes) |
    if (input.size() != sizeof(uint64_t)) {
        return false;
    }
    uint64_t size;
    memcpy(&size, input.data(), sizeof(size));
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, size);
    uint64_t remote_ptr = 0;
    uint64_t remote_size = 0;
    if (buffer != nullptr) {
        remote_ptr = reinterpret_cast<uint64_t>(buffer);
        remote_size = buffer->size;
        GGML_PRINT_DEBUG("[%s] size: %" PRIu64 " -> remote_ptr: %" PRIx64 ", remote_size: %" PRIu64 "\n", __func__, size, remote_ptr, remote_size);
        buffers.insert(buffer);
    } else {
        GGML_PRINT_DEBUG("[%s] size: %" PRIu64 " -> failed\n", __func__, size);
    }
    // output serialization format: | remote_ptr (8 bytes) | remote_size (8 bytes) |
    output.resize(2*sizeof(uint64_t), 0);
    memcpy(output.data(), &remote_ptr, sizeof(remote_ptr));
    memcpy(output.data() + sizeof(uint64_t), &remote_size, sizeof(remote_size));
    return true;
}

void rpc_server::get_alignment(std::vector<uint8_t> & output) {
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    GGML_PRINT_DEBUG("[%s] alignment: %lu\n", __func__, alignment);
    // output serialization format: | alignment (8 bytes) |
    output.resize(sizeof(uint64_t), 0);
    memcpy(output.data(), &alignment, sizeof(alignment));
}

void rpc_server::get_max_size(std::vector<uint8_t> & output) {
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    size_t max_size = ggml_backend_buft_get_max_size(buft);
    GGML_PRINT_DEBUG("[%s] max_size: %lu\n", __func__, max_size);
    // output serialization format: | max_size (8 bytes) |
    output.resize(sizeof(uint64_t), 0);
    memcpy(output.data(), &max_size, sizeof(max_size));
}

bool rpc_server::buffer_get_base(const std::vector<uint8_t> & input, std::vector<uint8_t> & output) {
    // input serialization format: | remote_ptr (8 bytes) |
    if (input.size() != sizeof(uint64_t)) {
        return false;
    }
    uint64_t remote_ptr;
    memcpy(&remote_ptr, input.data(), sizeof(remote_ptr));
    GGML_PRINT_DEBUG("[%s] remote_ptr: %" PRIx64 "\n", __func__, remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_PRINT_DEBUG("[%s] buffer not found\n", __func__);
        return false;
    }
    void * base = ggml_backend_buffer_get_base(buffer);
    // output serialization format: | base_ptr (8 bytes) |
    uint64_t base_ptr = reinterpret_cast<uint64_t>(base);
    output.resize(sizeof(uint64_t), 0);
    memcpy(output.data(), &base_ptr, sizeof(base_ptr));
    return true;
}

bool rpc_server::free_buffer(const std::vector<uint8_t> & input) {
    // input serialization format: | remote_ptr (8 bytes) |
    if (input.size() != sizeof(uint64_t)) {
        return false;
    }
    uint64_t remote_ptr;
    memcpy(&remote_ptr, input.data(), sizeof(remote_ptr));
    GGML_PRINT_DEBUG("[%s] remote_ptr: %" PRIx64 "\n", __func__, remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_PRINT_DEBUG("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_free(buffer);
    buffers.erase(buffer);
    return true;
}

bool rpc_server::buffer_clear(const std::vector<uint8_t> & input) {
    // input serialization format: | remote_ptr (8 bytes) | value (1 byte) |
    if (input.size() != sizeof(uint64_t) + sizeof(uint8_t)) {
        return false;
    }
    uint64_t remote_ptr;
    memcpy(&remote_ptr, input.data(), sizeof(remote_ptr));
    uint8_t value;
    memcpy(&value, input.data() + sizeof(uint64_t), sizeof(value));
    GGML_PRINT_DEBUG("[%s] remote_ptr: %" PRIx64 ", value: %u\n", __func__, remote_ptr, value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_PRINT_DEBUG("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_clear(buffer, value);
    return true;
}

ggml_tensor * rpc_server::deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor) {
    ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    if (result->buffer && buffers.find(result->buffer) == buffers.end()) {
        return nullptr;
    }
    result->op = (ggml_op) tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data = reinterpret_cast<void *>(tensor->data);
    ggml_set_name(result, tensor->name);
    return result;
}


bool rpc_server::set_tensor(const std::vector<uint8_t> & input) {
    // serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes) |
    if (input.size() < sizeof(rpc_tensor) + sizeof(uint64_t)) {
        return false;
    }
    const rpc_tensor * in_tensor = (const rpc_tensor *)input.data();
    uint64_t offset;
    memcpy(&offset, input.data() + sizeof(rpc_tensor), sizeof(offset));
    size_t size = input.size() - sizeof(rpc_tensor) - sizeof(offset);

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr) {
        GGML_PRINT_DEBUG("[%s] error deserializing tensor\n", __func__);
        ggml_free(ctx);
        return false;
    }
    GGML_PRINT_DEBUG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %zu\n", __func__, (void*)tensor->buffer, tensor->data, offset, size);
    const void * data = input.data() + sizeof(rpc_tensor) + sizeof(offset);
    ggml_backend_tensor_set(tensor, data, offset, size);
    ggml_free(ctx);
    return true;
}

bool rpc_server::get_tensor(const std::vector<uint8_t> & input, std::vector<uint8_t> & output) {
    // serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) |
    if (input.size() != sizeof(rpc_tensor) + 2*sizeof(uint64_t)) {
        return false;
    }
    const rpc_tensor * in_tensor = (const rpc_tensor *)input.data();
    uint64_t offset;
    memcpy(&offset, input.data() + sizeof(rpc_tensor), sizeof(offset));
    uint64_t size;
    memcpy(&size, input.data() + sizeof(rpc_tensor) + sizeof(offset), sizeof(size));

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr) {
        GGML_PRINT_DEBUG("[%s] error deserializing tensor\n", __func__);
        ggml_free(ctx);
        return false;
    }
    GGML_PRINT_DEBUG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %" PRIu64 "\n", __func__, (void*)tensor->buffer, tensor->data, offset, size);
    // output serialization format: | data (size bytes) |
    output.resize(size, 0);
    ggml_backend_tensor_get(tensor, output.data(), offset, size);
    ggml_free(ctx);
    return true;
}

bool rpc_server::copy_tensor(const std::vector<uint8_t> & input, std::vector<uint8_t> & output) {
    // serialization format: | rpc_tensor src | rpc_tensor dst |
    if (input.size() != 2*sizeof(rpc_tensor)) {
        return false;
    }
    const rpc_tensor * rpc_src = (const rpc_tensor *)input.data();
    const rpc_tensor * rpc_dst = (const rpc_tensor *)(input.data() + sizeof(rpc_src));

    struct ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor * src = deserialize_tensor(ctx, rpc_src);
    ggml_tensor * dst = deserialize_tensor(ctx, rpc_dst);
    if (src == nullptr || dst == nullptr) {
        GGML_PRINT_DEBUG("[%s] error deserializing tensors\n", __func__);
        ggml_free(ctx);
        return false;
    }
    GGML_PRINT_DEBUG("[%s] src->buffer: %p, dst->buffer: %p\n", __func__, (void*)src->buffer, (void*)dst->buffer);
    bool result = ggml_backend_buffer_copy_tensor(src, dst);
    // output serialization format: | result (1 byte) |
    output.resize(1, 0);
    output[0] = result;
    ggml_free(ctx);
    return true;
}

void rpc_server::remote_set_tensor(std::shared_ptr<socket_t> sock, const rpc_tensor * rpc_src, const rpc_tensor * rpc_dst) {
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor * src = deserialize_tensor(ctx, rpc_src);
    size_t src_size = ggml_nbytes(src);

    // input serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes) |
    size_t offset = 0;
    size_t input_size = sizeof(rpc_tensor) + sizeof(uint64_t) + src_size;
    std::vector<uint8_t> input(input_size, 0);
    memcpy(input.data(), rpc_dst, sizeof(rpc_tensor));
    memcpy(input.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
    ggml_backend_tensor_get(src, input.data() + sizeof(rpc_tensor) + sizeof(offset), offset, src_size);
    std::vector<uint8_t> output;
    bool status = send_rpc_cmd(sock, SET_TENSOR, input, output);
    GGML_ASSERT(status);
    ggml_free(ctx);
}

bool rpc_server::remote_copy_tensor(const std::vector<uint8_t> & input, std::vector<uint8_t> & output) {
    // serialization format: | rpc_tensor src | rpc_tensor dst | dst_endpoint_size (4 bytes) | dst_endpoint (dst_endpoint_size bytes) |
    if (input.size() < 2*sizeof(rpc_tensor) + sizeof(uint32_t)) {
        return false;
    }
    const rpc_tensor * rpc_src = (const rpc_tensor *)input.data();
    const rpc_tensor * rpc_dst = (const rpc_tensor *)(input.data() + sizeof(rpc_tensor));
    uint32_t dst_endpoint_size;
    memcpy(&dst_endpoint_size, input.data() + 2*sizeof(rpc_tensor), sizeof(dst_endpoint_size));
    if (input.size() != 2*sizeof(rpc_tensor) + sizeof(uint32_t) + dst_endpoint_size) {
        return false;
    }
    // output serialization format: | result (1 byte) |
    output.resize(1, 0);

    const char * dst_endpoint_ptr = (const char *)(input.data() + 2*sizeof(rpc_tensor) + sizeof(uint32_t));
    std::string dst_endpoint(dst_endpoint_ptr, dst_endpoint_size);

    std::string host;
    int port;
    if (!parse_endpoint(dst_endpoint, host, port)) {
        output[0] = false;
        return true;
    }
    auto sock = socket_connect(host.c_str(), port);
    if (sock == nullptr) {
        output[0] = false;
        return true;
    }
    send_hello(sock, SERVER);
    remote_set_tensor(sock, rpc_src, rpc_dst);
    output.resize(1, 0);
    output[0] = true;
    return true;
}

ggml_tensor * rpc_server::create_node(uint64_t id,
                                      struct ggml_context * ctx,
                                      const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map) {
    if (id == 0) {
        return nullptr;
    }
    if (tensor_map.find(id) != tensor_map.end()) {
        return tensor_map[id];
    }
    const rpc_tensor * tensor = tensor_ptrs.at(id);
    struct ggml_tensor * result = deserialize_tensor(ctx, tensor);
    if (result == nullptr) {
        return nullptr;
    }
    tensor_map[id] = result;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        result->src[i] = create_node(tensor->src[i], ctx, tensor_ptrs, tensor_map);
    }
    result->view_src = create_node(tensor->view_src, ctx, tensor_ptrs, tensor_map);
    result->view_offs = tensor->view_offs;
    return result;
}

bool rpc_server::graph_compute(const std::vector<uint8_t> & input, std::vector<uint8_t> & output) {
    // serialization format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    if (input.size() < sizeof(uint32_t)) {
        return false;
    }
    uint32_t n_nodes;
    memcpy(&n_nodes, input.data(), sizeof(n_nodes));
    if (input.size() < sizeof(uint32_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t)) {
        return false;
    }
    const uint64_t * nodes = (const uint64_t *)(input.data() + sizeof(n_nodes));
    uint32_t n_tensors;
    memcpy(&n_tensors, input.data() + sizeof(n_nodes) + n_nodes*sizeof(uint64_t), sizeof(n_tensors));
    if (input.size() < sizeof(uint32_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t) + n_tensors*sizeof(rpc_tensor)) {
        return false;
    }
    const rpc_tensor * tensors = (const rpc_tensor *)(input.data() + sizeof(n_nodes) + n_nodes*sizeof(uint64_t) + sizeof(n_tensors));
    GGML_PRINT_DEBUG("[%s] n_nodes: %u, n_tensors: %u\n", __func__, n_nodes, n_tensors);

    static size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, false);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, const rpc_tensor*> tensor_ptrs;
    for (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs[tensors[i].id] = &tensors[i];
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    for (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], sizeof(id));
        graph->nodes[i] = create_node(id, ctx, tensor_ptrs, tensor_map);
    }
    ggml_status status = ggml_backend_graph_compute(backend, graph);
    // output serialization format: | status (1 byte) |
    output.resize(1, 0);
    output[0] = status;
    ggml_free(ctx);
    return true;
}

void rpc_server::free_all_buffers() {
    for (auto buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
    buffers.clear();
}

rpc_server::~rpc_server() {
    free_all_buffers();
}

static void process_requests(ggml_backend_t backend, request_queue_ptr requestq) {
    rpc_server server(backend);
    while (true) {
        rpc_request_ptr request;
        requestq->pop(&request);
        rpc_response_ptr response = std::make_shared<rpc_response>();
        bool ok = true;
        switch (request->cmd) {
            case ALLOC_BUFFER: {
                ok = server.alloc_buffer(request->input, response->output);
                break;
            }
            case GET_ALIGNMENT: {
                server.get_alignment(response->output);
                break;
            }
            case GET_MAX_SIZE: {
                server.get_max_size(response->output);
                break;
            }
            case BUFFER_GET_BASE: {
                ok = server.buffer_get_base(request->input, response->output);
                break;
            }
            case FREE_BUFFER: {
                ok = server.free_buffer(request->input);
                break;
            }
            case BUFFER_CLEAR: {
                ok = server.buffer_clear(request->input);
                break;
            }
            case SET_TENSOR: {
                ok = server.set_tensor(request->input);
                break;
            }
            case GET_TENSOR: {
                ok = server.get_tensor(request->input, response->output);
                break;
            }
            case COPY_TENSOR: {
                ok = server.copy_tensor(request->input, response->output);
                break;
            }
            case REMOTE_COPY_TENSOR: {
                ok = server.remote_copy_tensor(request->input, response->output);
                break;
            }
            case GRAPH_COMPUTE: {
                ok = server.graph_compute(request->input, response->output);
                break;
            }
            case GET_DEVICE_MEMORY: {
                break;
            }
            case FREE_ALL_BUFFERS: {
                server.free_all_buffers();
                continue;
            }
            default: {
                fprintf(stderr, "Unknown command: %d\n", request->cmd);
                ok = false;
            }
        }
        response->status = ok;
        request->response_queue->push(response);
    }
}

static bool recv_rpc_cmd(sockfd_t sockfd, rpc_cmd & cmd, std::vector<uint8_t> & input) {
    uint8_t cmd_u8;
    if (!recv_data(sockfd, &cmd_u8, 1)) {
        return false;
    }
    cmd = (rpc_cmd)cmd_u8;
    uint64_t input_size;
    if (!recv_data(sockfd, &input_size, sizeof(input_size))) {
        return false;
    }
    input.resize(input_size);
    if (!recv_data(sockfd, input.data(), input_size)) {
        return false;
    }
    return true;
}

static void rpc_serve_client(request_queue_ptr requestq, std::shared_ptr<socket_t> sock, size_t free_mem, size_t total_mem) {
    auto responseq = std::make_shared<response_queue>();
    while (true) {
        auto request = std::make_shared<rpc_request>();
        if (!recv_rpc_cmd(sock->fd, request->cmd, request->input)) {
            break;
        }
        request->response_queue = responseq;
        bool ok = true;
        auto response = std::make_shared<rpc_response>();
        switch (request->cmd) {
            case ALLOC_BUFFER:
            case GET_ALIGNMENT:
            case GET_MAX_SIZE:
            case BUFFER_GET_BASE:
            case FREE_BUFFER:
            case BUFFER_CLEAR:
            case SET_TENSOR:
            case GET_TENSOR:
            case COPY_TENSOR:
            case REMOTE_COPY_TENSOR:
            case GRAPH_COMPUTE: {
                requestq->push(request);
                responseq->pop(&response);
                break;
            }
            case GET_DEVICE_MEMORY: {
                // output serialization format: | free (8 bytes) | total (8 bytes) |
                response->output.resize(2*sizeof(uint64_t), 0);
                memcpy(response->output.data(), &free_mem, sizeof(free_mem));
                memcpy(response->output.data() + sizeof(uint64_t), &total_mem, sizeof(total_mem));
                break;
            }
            default: {
                fprintf(stderr, "Unexpected command: %d\n", request->cmd);
                ok = false;
            }
        }
        if (!ok) {
            break;
        }
        uint64_t output_size = response->output.size();
        if (!send_data(sock->fd, &output_size, sizeof(output_size))) {
            break;
        }
        if (!send_data(sock->fd, response->output.data(), output_size)) {
            break;
        }
    }
    auto request = std::make_shared<rpc_request>();
    request->cmd = FREE_ALL_BUFFERS;
    requestq->push(request);
}

static void rpc_serve_server(request_queue_ptr requestq, std::shared_ptr<socket_t> sock) {
    auto responseq = std::make_shared<response_queue>();
    auto request = std::make_shared<rpc_request>();
    if (!recv_rpc_cmd(sock->fd, request->cmd, request->input)) {
        return;
    }
    if (request->cmd != SET_TENSOR) {
        fprintf(stderr, "Unexpected command: %d\n", request->cmd);
        return;
    }
    request->response_queue = responseq;
    auto response = std::make_shared<rpc_response>();
    requestq->push(request);
    responseq->pop(&response);
    uint64_t output_size = response->output.size();
    if (!send_data(sock->fd, &output_size, sizeof(output_size))) {
        return;
    }
    send_data(sock->fd, response->output.data(), output_size);
}

static bool recv_hello(std::shared_ptr<socket_t> sock, rpc_actor & actor) {
    rpc_cmd cmd;
    std::vector<uint8_t> input;
    if (!recv_rpc_cmd(sock->fd, cmd, input)) {
        return false;
    }
    if (cmd != HELLO || input.size() != 1) {
        return false;
    }
    if (input[0] != CLIENT && input[0] != SERVER) {
        return false;
    }
    actor = (rpc_actor)input[0];
    uint64_t output_size = 0;
    if (!send_data(sock->fd, &output_size, sizeof(output_size))) {
        return false;
    }
    return true;
}

static std::mutex client_mutex;
static std::mutex server_mutex;

void start_rpc_server(ggml_backend_t backend, const char * endpoint, size_t free_mem, size_t total_mem) {
#ifndef _WIN32
    // prevent SIGPIPE when writing to closed socket
    signal(SIGPIPE, SIG_IGN);
#endif
    auto requestq = std::make_shared<request_queue>();
    std::thread backend_thread = std::thread([=] {
        process_requests(backend, requestq);
    });

    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        return;
    }
#ifdef _WIN32
    {
        WSADATA wsaData;
        int res = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if (res != 0) {
            fprintf(stderr, "WSAStartup failed: %d\n", res);
            return;
        }
    }
#endif
    auto server_socket = create_server_socket(host.c_str(), port);
    if (server_socket == nullptr) {
        fprintf(stderr, "Failed to create server socket\n");
        return;
    }
    while (true) {
        auto client_socket = socket_accept(server_socket->fd);
        if (client_socket == nullptr) {
            fprintf(stderr, "Failed to accept client connection\n");
            return;
        }
        rpc_actor actor;
        if (!recv_hello(client_socket, actor)) {
            continue;
        }
        if (actor == CLIENT) {
            std::thread client_thread = std::thread([=] {
                std::lock_guard<std::mutex> lock(client_mutex);
                printf("Accepted client connection, free_mem=%zu, total_mem=%zu\n", free_mem, total_mem);
                rpc_serve_client(requestq, client_socket, free_mem, total_mem);
                printf("Client connection closed\n");
            });
            client_thread.detach();
        } else {
            std::thread server_thread = std::thread([=] {
                std::lock_guard<std::mutex> lock(server_mutex);
                printf("Accepted connection from another server\n");
                rpc_serve_server(requestq, client_socket);
                printf("Server connection closed\n");
            });
            server_thread.detach();
        }
    }
#ifdef _WIN32
    WSACleanup();
#endif
}
