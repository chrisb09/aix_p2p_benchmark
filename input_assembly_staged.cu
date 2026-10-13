#include <cuda_runtime.h>
#include <mpi.h>

#ifdef WITH_TORCH_MODEL
#include <ATen/Parallel.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/script.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kCreditTag = 17001;
constexpr int kInputTag = 17002;
constexpr int kInputAckTag = 17004;
constexpr int kDefaultPayloadBytes = 6'220'800;  // 320 x 270 x 18 FP32 values.
constexpr int kOutputRatio = 18;                 // WaterCNN: 18 FP32 input values -> 1 FP32 output value.
constexpr int kRecordWidth = 10;

enum class Mode { Collective, P2PCredit1, P2PFullCredits };
enum class Stage { PinnedMPI, H2D, H2DCompute, Full };
enum class MpiThreadRequest { Single, Multiple };
enum class GroupLayout { Contiguous, Strided };

struct Options {
    Mode mode = Mode::Collective;
    Stage stage = Stage::PinnedMPI;
    int group_size = 24;
    int payload_bytes = kDefaultPayloadBytes;
    int warmups = 20;
    int iterations = 100;
    int pipeline_depth = 3;
    // 40k repetitions calibrate to approximately 12.5 ms controller-local
    // forward work, matching 50 ms summed WaterCNN forwards across 4 controllers.
    int compute_repeats = 40000;
    bool async_controller = false;
    MpiThreadRequest mpi_thread_request = MpiThreadRequest::Multiple;
    GroupLayout group_layout = GroupLayout::Contiguous;
    int submit_cpu_us = 0;
    std::string model = "mock";
    std::string output = "input_assembly_staged.csv";
};

struct StageRequest {
    cudaEvent_t h2d_start{};
    cudaEvent_t h2d_end{};
    cudaEvent_t compute_start{};
    cudaEvent_t compute_end{};
    cudaEvent_t d2h_start{};
    cudaEvent_t d2h_end{};
    cudaEvent_t complete{};
#ifdef WITH_TORCH_MODEL
    torch::Tensor output_keepalive;
#endif
};

__global__ void mock_inference_kernel(const float* input, float* output, size_t output_count, int repeats) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= output_count) {
        return;
    }
    float value = input[index * kOutputRatio];
    for (int repeat = 0; repeat < repeats; ++repeat) {
        value = value * 1.000001f + 0.000001f;
    }
    output[index] = value;
}

const char* mode_name(Mode mode) {
    switch (mode) {
        case Mode::Collective: return "collective_gatherv";
        case Mode::P2PCredit1: return "p2p_credit1";
        case Mode::P2PFullCredits: return "p2p_fullcredits";
    }
    return "unknown";
}

const char* stage_name(Stage stage) {
    switch (stage) {
        case Stage::PinnedMPI: return "pinned_mpi";
        case Stage::H2D: return "h2d";
        case Stage::H2DCompute: return "h2d_compute";
        case Stage::Full: return "h2d_compute_d2h";
    }
    return "unknown";
}

const char* mpi_thread_request_name(MpiThreadRequest request) {
    return request == MpiThreadRequest::Single ? "single" : "multiple";
}

const char* group_layout_name(GroupLayout layout) {
    return layout == GroupLayout::Strided ? "strided" : "contiguous";
}

[[noreturn]] void usage_and_exit(const char* message = nullptr) {
    if (message != nullptr) {
        std::cerr << "Error: " << message << "\n\n";
    }
    std::cerr << "Usage: aix_input_assembly_staged --mode collective|p2p-credit1|p2p-fullcredits "
                 "--stage pinned-mpi|h2d|h2d-compute|full [--group-size 24] "
                 "[--payload-bytes 6220800] [--warmups 20] [--iterations 100] "
                 "[--pipeline-depth 3] [--compute-repeats 40000] [--async-controller 0|1] "
                 "[--mpi-thread single|multiple] [--submit-cpu-us N] [--group-layout contiguous|strided] "
                 "[--model mock|MODEL.pt] [--output FILE]\n";
    std::exit(message == nullptr ? EXIT_SUCCESS : EXIT_FAILURE);
}

int parse_positive_int(const std::string& value, const char* flag) {
    try {
        const long long parsed = std::stoll(value);
        if (parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
            throw std::out_of_range("outside int range");
        }
        return static_cast<int>(parsed);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string(flag) + " requires a positive integer");
    }
}

int parse_nonnegative_int(const std::string& value, const char* flag) {
    try {
        const long long parsed = std::stoll(value);
        if (parsed < 0 || parsed > std::numeric_limits<int>::max()) {
            throw std::out_of_range("outside int range");
        }
        return static_cast<int>(parsed);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string(flag) + " requires a non-negative integer");
    }
}

bool parse_bool(const std::string& value, const char* flag) {
    if (value == "0") return false;
    if (value == "1") return true;
    throw std::runtime_error(std::string(flag) + " must be 0 or 1");
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--help" || arg == "-h") {
            usage_and_exit();
        }
        if (index + 1 >= argc) {
            throw std::runtime_error("missing value for " + arg);
        }
        const std::string value = argv[++index];
        if (arg == "--mode") {
            if (value == "collective") options.mode = Mode::Collective;
            else if (value == "p2p-credit1") options.mode = Mode::P2PCredit1;
            else if (value == "p2p-fullcredits") options.mode = Mode::P2PFullCredits;
            else throw std::runtime_error("invalid --mode");
        } else if (arg == "--stage") {
            if (value == "pinned-mpi") options.stage = Stage::PinnedMPI;
            else if (value == "h2d") options.stage = Stage::H2D;
            else if (value == "h2d-compute") options.stage = Stage::H2DCompute;
            else if (value == "full") options.stage = Stage::Full;
            else throw std::runtime_error("invalid --stage");
        } else if (arg == "--group-size") options.group_size = parse_positive_int(value, "--group-size");
        else if (arg == "--payload-bytes") options.payload_bytes = parse_positive_int(value, "--payload-bytes");
        else if (arg == "--warmups") options.warmups = parse_positive_int(value, "--warmups");
        else if (arg == "--iterations") options.iterations = parse_positive_int(value, "--iterations");
        else if (arg == "--pipeline-depth") options.pipeline_depth = parse_positive_int(value, "--pipeline-depth");
        else if (arg == "--compute-repeats") options.compute_repeats = parse_positive_int(value, "--compute-repeats");
        else if (arg == "--async-controller") options.async_controller = parse_bool(value, "--async-controller");
        else if (arg == "--mpi-thread") {
            if (value == "single") options.mpi_thread_request = MpiThreadRequest::Single;
            else if (value == "multiple") options.mpi_thread_request = MpiThreadRequest::Multiple;
            else throw std::runtime_error("--mpi-thread must be single or multiple");
        }
        else if (arg == "--submit-cpu-us") options.submit_cpu_us = parse_nonnegative_int(value, "--submit-cpu-us");
        else if (arg == "--group-layout") {
            if (value == "contiguous") options.group_layout = GroupLayout::Contiguous;
            else if (value == "strided") options.group_layout = GroupLayout::Strided;
            else throw std::runtime_error("--group-layout must be contiguous or strided");
        }
        else if (arg == "--model") options.model = value;
        else if (arg == "--output") options.output = value;
        else throw std::runtime_error("unknown argument: " + arg);
    }
    return options;
}

void burn_controller_cpu(int microseconds) {
    if (microseconds == 0) return;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(microseconds);
    volatile uint64_t accumulator = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        accumulator = accumulator * 1664525ULL + 1013904223ULL;
    }
}

void check_mpi(int result, const char* operation, MPI_Comm comm) {
    if (result == MPI_SUCCESS) return;
    char error[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(result, error, &length);
    std::cerr << operation << " failed: " << std::string(error, length) << '\n';
    MPI_Abort(comm, result);
}

void check_cuda(cudaError_t result, const char* operation, MPI_Comm comm) {
    if (result == cudaSuccess) return;
    std::cerr << operation << " failed: " << cudaGetErrorString(result) << '\n';
    MPI_Abort(comm, EXIT_FAILURE);
}

void destroy_request_events(StageRequest& request) {
    cudaEventDestroy(request.h2d_start);
    cudaEventDestroy(request.h2d_end);
    cudaEventDestroy(request.compute_start);
    cudaEventDestroy(request.compute_end);
    cudaEventDestroy(request.d2h_start);
    cudaEventDestroy(request.d2h_end);
    cudaEventDestroy(request.complete);
}

void write_results(const Options& options, int world_size, int thread_level, const std::vector<double>& all_rows) {
    std::ofstream stream(options.output);
    if (!stream) throw std::runtime_error("cannot open output file: " + options.output);
    stream << "# mode=" << mode_name(options.mode) << ",stage=" << stage_name(options.stage)
           << ",world_size=" << world_size << ",group_size=" << options.group_size
           << ",workgroups=" << world_size / options.group_size
           << ",payload_bytes_per_rank=" << options.payload_bytes
           << ",pipeline_depth=" << options.pipeline_depth << ",compute_repeats=" << options.compute_repeats
           << ",async_controller=" << options.async_controller
           << ",mpi_thread_requested=" << mpi_thread_request_name(options.mpi_thread_request)
           << ",submit_cpu_us=" << options.submit_cpu_us
           << ",model=" << options.model
           << ",group_layout=" << group_layout_name(options.group_layout)
           << ",mpi_thread_level_provided=" << thread_level << "\n";
    stream << "mode,stage,async_controller,mpi_thread_requested,submit_cpu_us,model,group_layout,scenario_world_ranks,group_size,iteration,workgroup,input_ready_ms,stage_complete_ms,"
              "h2d_sum_ms,compute_sum_ms,d2h_sum_ms,scheduled_ranges,submit_cpu_sum_ms,submit_cpu_max_ms\n";
    stream << std::fixed << std::setprecision(6);
    for (int rank = 0; rank < world_size; ++rank) {
        for (int iteration = 0; iteration < options.iterations; ++iteration) {
            const size_t base = (static_cast<size_t>(rank) * options.iterations + iteration) * kRecordWidth;
            if (std::isnan(all_rows[base])) continue;
            stream << mode_name(options.mode) << ',' << stage_name(options.stage) << ',' << options.async_controller << ','
                   << mpi_thread_request_name(options.mpi_thread_request) << ',' << options.submit_cpu_us << ','
                   << options.model << ',' << group_layout_name(options.group_layout) << ',' << world_size << ','
                   << options.group_size << ',' << static_cast<int>(all_rows[base]) << ','
                   << static_cast<int>(all_rows[base + 1]) << ',' << all_rows[base + 2] << ','
                   << all_rows[base + 3] << ',' << all_rows[base + 4] << ',' << all_rows[base + 5] << ','
                   << all_rows[base + 6] << ',' << all_rows[base + 7] << ',' << all_rows[base + 8] << ','
                   << all_rows[base + 9] << '\n';
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    try {
        options = parse_options(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Argument error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    int provided_thread_level = MPI_THREAD_SINGLE;
    const int requested_thread_level = options.mpi_thread_request == MpiThreadRequest::Single
        ? MPI_THREAD_SINGLE : MPI_THREAD_MULTIPLE;
    MPI_Init_thread(&argc, &argv, requested_thread_level, &provided_thread_level);

    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    try {
        if (options.mode != Mode::Collective && options.async_controller && provided_thread_level < MPI_THREAD_MULTIPLE) {
            throw std::runtime_error("--async-controller 1 requires MPI_THREAD_MULTIPLE");
        }
        if (world_size % options.group_size != 0 || options.payload_bytes % (sizeof(float) * kOutputRatio) != 0) {
            throw std::runtime_error("world size/group size or payload/output ratio is invalid");
        }

        const int num_groups = world_size / options.group_size;
        const int workgroup = options.group_layout == GroupLayout::Strided
            ? world_rank % num_groups
            : world_rank / options.group_size;
        MPI_Comm group_comm = MPI_COMM_NULL;
        check_mpi(MPI_Comm_split(MPI_COMM_WORLD, workgroup, world_rank, &group_comm), "MPI_Comm_split", MPI_COMM_WORLD);
        int group_rank = 0;
        MPI_Comm_rank(group_comm, &group_rank);

        const size_t input_bytes = static_cast<size_t>(options.payload_bytes);
        const size_t output_bytes = input_bytes / kOutputRatio;
        const size_t output_floats = output_bytes / sizeof(float);
        std::vector<unsigned char> worker_input(input_bytes, static_cast<unsigned char>(world_rank));
        std::vector<int> counts(options.group_size, options.payload_bytes);
        std::vector<int> displacements(options.group_size);
        for (int rank = 0; rank < options.group_size; ++rank) displacements[rank] = rank * options.payload_bytes;

        unsigned char* controller_input = nullptr;
        unsigned char* controller_output = nullptr;
        float* device_input = nullptr;
        float* device_output = nullptr;
        cudaStream_t h2d_stream{};
        cudaStream_t compute_stream{};
        cudaStream_t d2h_stream{};
        int controller_device = 0;
        if (group_rank == 0) {
            int device_count = 0;
            check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount", group_comm);
            controller_device = workgroup % device_count;
            check_cuda(cudaSetDevice(controller_device), "cudaSetDevice", group_comm);
            check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&controller_input), input_bytes * options.group_size,
                                    cudaHostAllocDefault), "cudaHostAlloc input", group_comm);
            check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&controller_output), output_bytes * options.group_size,
                                    cudaHostAllocDefault), "cudaHostAlloc output", group_comm);
            std::memcpy(controller_input, worker_input.data(), input_bytes);
            if (options.stage != Stage::PinnedMPI) {
                check_cuda(cudaMalloc(&device_input, input_bytes * options.group_size), "cudaMalloc input", group_comm);
                check_cuda(cudaMalloc(&device_output, output_bytes * options.group_size), "cudaMalloc output", group_comm);
                check_cuda(cudaStreamCreateWithFlags(&h2d_stream, cudaStreamNonBlocking), "cudaStreamCreate h2d", group_comm);
                check_cuda(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking), "cudaStreamCreate compute", group_comm);
                check_cuda(cudaStreamCreateWithFlags(&d2h_stream, cudaStreamNonBlocking), "cudaStreamCreate d2h", group_comm);
            }
        }

#ifdef WITH_TORCH_MODEL
        std::shared_ptr<torch::jit::Module> torch_module;
#endif
        if (group_rank == 0 && options.model != "mock") {
#ifdef WITH_TORCH_MODEL
            if (options.stage == Stage::PinnedMPI) {
                throw std::runtime_error("--model requires a GPU stage (h2d, h2d-compute, or full)");
            }
            at::set_num_threads(1);
            try {
                torch_module = std::make_shared<torch::jit::Module>(torch::jit::load(options.model));
                torch_module->to(torch::Device(torch::kCUDA, controller_device));
                torch_module->eval();
            } catch (const c10::Error& error) {
                throw std::runtime_error("failed to load TorchScript model " + options.model + ": " + error.what());
            }
            std::cout << "Loaded TorchScript model: " << options.model << '\n';
#else
            throw std::runtime_error("--model requires a build with WITH_TORCH_MODEL");
#endif
        }

        std::vector<double> local_rows(static_cast<size_t>(options.iterations) * kRecordWidth,
                                       std::numeric_limits<double>::quiet_NaN());
        const int total_iterations = options.warmups + options.iterations;

        for (int iteration = 0; iteration < total_iterations; ++iteration) {
            std::vector<MPI_Request> input_requests(options.group_size, MPI_REQUEST_NULL);
            std::vector<MPI_Request> credit_requests(options.group_size, MPI_REQUEST_NULL);
            std::vector<MPI_Request> ack_requests(options.group_size, MPI_REQUEST_NULL);
            std::vector<int> control_values(options.group_size, 1);
            if (group_rank == 0 && options.mode != Mode::Collective) {
                for (int rank = 1; rank < options.group_size; ++rank) {
                    check_mpi(MPI_Irecv(controller_input + static_cast<size_t>(rank) * input_bytes, options.payload_bytes,
                                        MPI_BYTE, rank, kInputTag, group_comm, &input_requests[rank]),
                              "MPI_Irecv", group_comm);
                }
            }
            check_mpi(MPI_Barrier(group_comm), "MPI_Barrier", group_comm);

            double input_ready_ms = 0.0;
            double stage_complete_ms = 0.0;
            double h2d_sum_ms = 0.0;
            double compute_sum_ms = 0.0;
            double d2h_sum_ms = 0.0;
            double submit_cpu_sum_ms = 0.0;
            double submit_cpu_max_ms = 0.0;
            int scheduled_ranges = 0;

            if (group_rank != 0) {
                if (options.mode == Mode::Collective) {
                    check_mpi(MPI_Gatherv(worker_input.data(), options.payload_bytes, MPI_BYTE, nullptr, nullptr, nullptr,
                                          MPI_BYTE, 0, group_comm), "MPI_Gatherv", group_comm);
                } else {
                    int value = 0;
                    check_mpi(MPI_Recv(&value, 1, MPI_INT, 0, kCreditTag, group_comm, MPI_STATUS_IGNORE),
                              "MPI_Recv credit", group_comm);
                    check_mpi(MPI_Send(worker_input.data(), options.payload_bytes, MPI_BYTE, 0, kInputTag, group_comm),
                              "MPI_Send input", group_comm);
                    check_mpi(MPI_Recv(&value, 1, MPI_INT, 0, kInputAckTag, group_comm, MPI_STATUS_IGNORE),
                              "MPI_Recv ACK", group_comm);
                }
            } else {
                const double step_start = MPI_Wtime();
                std::mutex queue_mutex;
                std::condition_variable queue_cv;
                std::deque<int> ready_ranks;
                std::atomic<int> received_count{0};
                std::atomic<bool> progress_done{false};
                std::atomic<double> last_input_ready_s{step_start};
                std::thread progress_thread;
                int next_rank = 1;
                std::function<void(int)> issue_grants;
                std::function<bool()> poll_progress;

                if (options.mode == Mode::Collective) {
                    check_mpi(MPI_Gatherv(worker_input.data(), options.payload_bytes, MPI_BYTE, controller_input,
                                          counts.data(), displacements.data(), MPI_BYTE, 0, group_comm),
                              "MPI_Gatherv", group_comm);
                    last_input_ready_s = MPI_Wtime();
                    for (int rank = 0; rank < options.group_size; ++rank) ready_ranks.push_back(rank);
                } else {
                    const int initial_credits = options.mode == Mode::P2PCredit1 ? 1 : options.group_size - 1;
                    issue_grants = [&](int requested) {
                        const int count = std::min(requested, options.group_size - next_rank);
                        for (int grant = 0; grant < count; ++grant) {
                            const int destination = next_rank++;
                            check_mpi(MPI_Isend(&control_values[destination], 1, MPI_INT, destination, kCreditTag,
                                                group_comm, &credit_requests[destination]), "MPI_Isend credit", group_comm);
                        }
                    };
                    poll_progress = [&]() {
                        std::vector<int> indices(options.group_size);
                        std::vector<MPI_Status> statuses(options.group_size);
                        int completed = 0;
                        check_mpi(MPI_Testsome(options.group_size, input_requests.data(), &completed,
                                               indices.data(), statuses.data()), "MPI_Testsome", group_comm);
                        if (completed == MPI_UNDEFINED || completed == 0) return false;
                        for (int index = 0; index < completed; ++index) {
                            const int source = indices[index];
                            check_mpi(MPI_Isend(&control_values[source], 1, MPI_INT, source, kInputAckTag,
                                                group_comm, &ack_requests[source]), "MPI_Isend ACK", group_comm);
                            if (options.mode == Mode::P2PCredit1) issue_grants(2);
                            {
                                std::lock_guard<std::mutex> lock(queue_mutex);
                                ready_ranks.push_back(source);
                            }
                            const int now_received = received_count.fetch_add(1, std::memory_order_relaxed) + 1;
                            if (now_received == options.group_size - 1) {
                                last_input_ready_s.store(MPI_Wtime(), std::memory_order_relaxed);
                                progress_done.store(true, std::memory_order_relaxed);
                            }
                            queue_cv.notify_one();
                        }
                        return true;
                    };
                    issue_grants(initial_credits);
                    ready_ranks.push_back(0);  // AIx marks the controller-local segment ready immediately.
                    if (options.async_controller) {
                        progress_thread = std::thread([&]() {
                        while (received_count.load(std::memory_order_relaxed) < options.group_size - 1) {
                            poll_progress();
                            std::this_thread::yield();
                        }
                        progress_done.store(true, std::memory_order_relaxed);
                        queue_cv.notify_one();
                        });
                    }
                }

                std::deque<StageRequest> active_requests;
                const auto retire_one = [&](bool wait) {
                    if (active_requests.empty()) return false;
                    StageRequest& request = active_requests.front();
                    if (wait) check_cuda(cudaEventSynchronize(request.complete), "cudaEventSynchronize", group_comm);
                    else if (cudaEventQuery(request.complete) == cudaErrorNotReady) return false;
                    float elapsed = 0.0F;
                    check_cuda(cudaEventElapsedTime(&elapsed, request.h2d_start, request.h2d_end), "cudaEventElapsedTime h2d", group_comm);
                    h2d_sum_ms += elapsed;
                    if (options.stage == Stage::H2DCompute || options.stage == Stage::Full) {
                        check_cuda(cudaEventElapsedTime(&elapsed, request.compute_start, request.compute_end),
                                   "cudaEventElapsedTime compute", group_comm);
                        compute_sum_ms += elapsed;
                    }
                    if (options.stage == Stage::Full) {
                        check_cuda(cudaEventElapsedTime(&elapsed, request.d2h_start, request.d2h_end),
                                   "cudaEventElapsedTime d2h", group_comm);
                        d2h_sum_ms += elapsed;
                    }
                    destroy_request_events(request);
                    active_requests.pop_front();
                    return true;
                };

                const auto submit_gpu_range = [&](int rank) {
                    if (options.stage != Stage::PinnedMPI) {
                        while (static_cast<int>(active_requests.size()) >= options.pipeline_depth) retire_one(true);
                    }
                    const auto submit_cpu_begin = std::chrono::steady_clock::now();
                    burn_controller_cpu(options.submit_cpu_us);
                    const auto record_submit_cpu = [&]() {
                        const double submit_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - submit_cpu_begin).count();
                        submit_cpu_sum_ms += submit_ms;
                        submit_cpu_max_ms = std::max(submit_cpu_max_ms, submit_ms);
                    };
                    if (options.stage == Stage::PinnedMPI) {
                        record_submit_cpu();
                        return;
                    }
                    StageRequest request;
                    check_cuda(cudaEventCreate(&request.h2d_start), "cudaEventCreate h2d start", group_comm);
                    check_cuda(cudaEventCreate(&request.h2d_end), "cudaEventCreate h2d end", group_comm);
                    check_cuda(cudaEventCreate(&request.compute_start), "cudaEventCreate compute start", group_comm);
                    check_cuda(cudaEventCreate(&request.compute_end), "cudaEventCreate compute end", group_comm);
                    check_cuda(cudaEventCreate(&request.d2h_start), "cudaEventCreate d2h start", group_comm);
                    check_cuda(cudaEventCreate(&request.d2h_end), "cudaEventCreate d2h end", group_comm);
                    check_cuda(cudaEventCreate(&request.complete), "cudaEventCreate complete", group_comm);
                    auto* input_host = controller_input + static_cast<size_t>(rank) * input_bytes;
                    auto* input_gpu = reinterpret_cast<unsigned char*>(device_input) + static_cast<size_t>(rank) * input_bytes;
                    auto* output_host = controller_output + static_cast<size_t>(rank) * output_bytes;
                    float* output_gpu = device_output + static_cast<size_t>(rank) * output_floats;
                    check_cuda(cudaEventRecord(request.h2d_start, h2d_stream), "cudaEventRecord h2d start", group_comm);
                    check_cuda(cudaMemcpyAsync(input_gpu, input_host, input_bytes, cudaMemcpyHostToDevice, h2d_stream),
                               "cudaMemcpyAsync H2D", group_comm);
                    check_cuda(cudaEventRecord(request.h2d_end, h2d_stream), "cudaEventRecord h2d end", group_comm);
                    if (options.stage == Stage::H2D) {
                        check_cuda(cudaEventRecord(request.complete, h2d_stream), "cudaEventRecord h2d complete", group_comm);
                    } else {
                        check_cuda(cudaStreamWaitEvent(compute_stream, request.h2d_end, 0), "cudaStreamWaitEvent H2D", group_comm);
                        check_cuda(cudaEventRecord(request.compute_start, compute_stream), "cudaEventRecord compute start", group_comm);
                        bool handled_by_model = false;
#ifdef WITH_TORCH_MODEL
                        if (torch_module) {
                            const int64_t samples = static_cast<int64_t>(output_floats);
                            auto tensor_options = torch::TensorOptions()
                                .dtype(torch::kFloat32).device(torch::kCUDA, controller_device);
                            auto input_tensor = torch::from_blob(
                                reinterpret_cast<float*>(input_gpu), {samples, kOutputRatio}, tensor_options);
                            c10::cuda::CUDAStreamGuard guard(
                                c10::cuda::getStreamFromExternal(compute_stream, controller_device));
                            torch::NoGradGuard no_grad;
                            auto output_tensor = torch_module->forward({input_tensor}).toTensor();
                            output_gpu = output_tensor.data_ptr<float>();
                            request.output_keepalive = output_tensor;
                            handled_by_model = true;
                        }
#endif
                        if (!handled_by_model) {
                            const int threads = 256;
                            const int blocks = static_cast<int>((output_floats + threads - 1) / threads);
                            mock_inference_kernel<<<blocks, threads, 0, compute_stream>>>(reinterpret_cast<float*>(input_gpu),
                                                                                         output_gpu, output_floats,
                                                                                         options.compute_repeats);
                            check_cuda(cudaGetLastError(), "mock_inference_kernel", group_comm);
                        }
                        check_cuda(cudaEventRecord(request.compute_end, compute_stream), "cudaEventRecord compute end", group_comm);
                        if (options.stage == Stage::H2DCompute) {
                            check_cuda(cudaEventRecord(request.complete, compute_stream), "cudaEventRecord compute complete", group_comm);
                        } else {
                            check_cuda(cudaStreamWaitEvent(d2h_stream, request.compute_end, 0), "cudaStreamWaitEvent compute", group_comm);
                            check_cuda(cudaEventRecord(request.d2h_start, d2h_stream), "cudaEventRecord d2h start", group_comm);
                            check_cuda(cudaMemcpyAsync(output_host, output_gpu, output_bytes, cudaMemcpyDeviceToHost, d2h_stream),
                                       "cudaMemcpyAsync D2H", group_comm);
                            check_cuda(cudaEventRecord(request.d2h_end, d2h_stream), "cudaEventRecord d2h end", group_comm);
                            check_cuda(cudaEventRecord(request.complete, d2h_stream), "cudaEventRecord d2h complete", group_comm);
                        }
                    }
                    active_requests.push_back(request);
                    ++scheduled_ranges;
                    record_submit_cpu();
                };

                int scheduled = 0;
                while (scheduled < options.group_size) {
                    int rank = -1;
                    {
                        std::lock_guard<std::mutex> lock(queue_mutex);
                        if (!ready_ranks.empty()) {
                            rank = ready_ranks.front();
                            ready_ranks.pop_front();
                        }
                    }
                    if (rank < 0 && options.mode != Mode::Collective && !options.async_controller) {
                        if (!poll_progress()) std::this_thread::yield();
                        continue;
                    }
                    if (rank < 0 && options.async_controller) {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        queue_cv.wait(lock, [&]() {
                            return !ready_ranks.empty() || progress_done.load(std::memory_order_relaxed);
                        });
                        if (ready_ranks.empty() && progress_done.load(std::memory_order_relaxed)) {
                            throw std::runtime_error("P2P progress ended before every rank became ready");
                        }
                        continue;
                    }
                    if (rank >= 0) {
                        submit_gpu_range(rank);
                        ++scheduled;
                    }
                    while (retire_one(false)) {}
                }
                while (retire_one(true)) {}
                if (progress_thread.joinable()) progress_thread.join();
                stage_complete_ms = (MPI_Wtime() - step_start) * 1.0e3;
                input_ready_ms = (last_input_ready_s.load(std::memory_order_relaxed) - step_start) * 1.0e3;

                if (options.mode != Mode::Collective) {
                    check_mpi(MPI_Waitall(options.group_size, credit_requests.data(), MPI_STATUSES_IGNORE),
                              "MPI_Waitall credits", group_comm);
                    check_mpi(MPI_Waitall(options.group_size, ack_requests.data(), MPI_STATUSES_IGNORE),
                              "MPI_Waitall ACKs", group_comm);
                }
            }

            check_mpi(MPI_Barrier(group_comm), "MPI_Barrier", group_comm);
            if (iteration >= options.warmups && group_rank == 0) {
                const size_t base = static_cast<size_t>(iteration - options.warmups) * kRecordWidth;
                local_rows[base] = static_cast<double>(iteration - options.warmups);
                local_rows[base + 1] = static_cast<double>(workgroup);
                local_rows[base + 2] = input_ready_ms;
                local_rows[base + 3] = stage_complete_ms;
                local_rows[base + 4] = h2d_sum_ms;
                local_rows[base + 5] = compute_sum_ms;
                local_rows[base + 6] = d2h_sum_ms;
                local_rows[base + 7] = static_cast<double>(scheduled_ranges);
                local_rows[base + 8] = submit_cpu_sum_ms;
                local_rows[base + 9] = submit_cpu_max_ms;
            }
        }

        std::vector<double> all_rows;
        if (world_rank == 0) all_rows.resize(static_cast<size_t>(world_size) * local_rows.size());
        check_mpi(MPI_Gather(local_rows.data(), static_cast<int>(local_rows.size()), MPI_DOUBLE,
                             all_rows.data(), static_cast<int>(local_rows.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD),
                  "MPI_Gather metrics", MPI_COMM_WORLD);
        if (world_rank == 0) {
            write_results(options, world_size, provided_thread_level, all_rows);
            std::cout << "Wrote " << options.output << " for " << mode_name(options.mode) << "/" << stage_name(options.stage) << '\n';
        }

        if (group_rank == 0) {
            if (h2d_stream) cudaStreamDestroy(h2d_stream);
            if (compute_stream) cudaStreamDestroy(compute_stream);
            if (d2h_stream) cudaStreamDestroy(d2h_stream);
            if (device_input) cudaFree(device_input);
            if (device_output) cudaFree(device_output);
            cudaFreeHost(controller_input);
            cudaFreeHost(controller_output);
        }
        MPI_Comm_free(&group_comm);
        MPI_Finalize();
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        if (world_rank == 0) std::cerr << "Fatal error: " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    return EXIT_FAILURE;
}
