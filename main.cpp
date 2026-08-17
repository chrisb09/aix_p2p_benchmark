#include "aixeleratorService/aixeleratorService.h"

#include <mpi.h>
#include <sys/resource.h>
#include <dlfcn.h>

#ifdef USE_SCOREP
#include <scorep/SCOREP_User.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
static double get_local_peak_rss_mb() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return static_cast<double>(usage.ru_maxrss) / 1024.0;
    }
    return 0.0;
}

static double get_gpu_vram_used_mb() {
    typedef int (*cuda_mem_info_fn)(size_t*, size_t*);
    static cuda_mem_info_fn fn = nullptr;
    static bool tried = false;
    if (!tried) {
        tried = true;
        void* handle = dlopen("libcudart.so", RTLD_LAZY | RTLD_GLOBAL);
        if (!handle) {
            handle = dlopen("libcudart.so.12", RTLD_LAZY | RTLD_GLOBAL);
        }
        if (handle) {
            fn = reinterpret_cast<cuda_mem_info_fn>(dlsym(handle, "cudaMemGetInfo"));
        }
    }
    if (fn) {
        size_t free_b = 0, total_b = 0;
        if (fn(&free_b, &total_b) == 0 && total_b > 0) {
            return static_cast<double>(total_b - free_b) / (1024.0 * 1024.0);
        }
    }
    return 0.0;
}

struct Options {
    std::string model;
    std::string output = "aix_p2p_steps.csv";
    std::string raw_output_prefix;
    int64_t samples_per_rank = 129600;
    int batch_size = 4000000;
    int warmup_steps = 3;
    int measured_steps = 10;
    int input_seed = 1337;
    std::vector<int64_t> input_shape = {129600, 18};
    std::vector<int64_t> output_shape = {129600};
};

struct TimelineMark {
    int step;
    double time_s;
    const char* event;
};

double corrected_time(double clock_offset)
{
    return MPI_Wtime() + clock_offset;
}

void write_solver_timeline(const char* directory, int rank, const std::vector<TimelineMark>& events)
{
    if (!directory || events.empty()) {
        return;
    }
    std::filesystem::create_directories(directory);
    const auto path = std::filesystem::path(directory) /
        ("aix_p2p_solver_timeline_rank_" + std::to_string(rank) + ".csv");
    std::ofstream stream(path);
    stream << "step,time_s,world_rank,workgroup_rank,is_controller,event,peer_workgroup_rank,"
           << "range_first_rank,range_end_rank,sample_start,sample_count\n";
    for (const auto& event : events) {
        stream << event.step << ',' << std::setprecision(17) << event.time_s << ',' << rank
               << ",-1,0," << event.event << ",-1,-1,-1,-1,-1\n";
    }
}

void write_timeline_metadata(const char* directory, const Options& options)
{
    if (!directory) {
        return;
    }
    std::filesystem::create_directories(directory);
    std::ofstream stream(std::filesystem::path(directory) / "timeline_metadata.json");
    stream << "{\n"
           << "  \"model\": \"" << std::filesystem::path(options.model).filename().string() << "\",\n"
           << "  \"warmup_steps\": " << options.warmup_steps << ",\n"
           << "  \"measured_steps\": " << options.measured_steps << ",\n"
           << "  \"batch_size\": " << options.batch_size << "\n"
           << "}\n";
}

[[noreturn]] void usage(const char* executable)
{
    std::cerr << "Usage: " << executable << " --model FILE [options]\n"
              << "  --samples-per-rank N   Default: 129600\n"
              << "  --input-shape S1,S2..  Default: <samples_per_rank>,18\n"
              << "  --output-shape S1,S2. Default: <samples_per_rank>\n"
              << "  --batch-size N         Default: 4000000\n"
              << "  --warmup-steps N       Default: 3\n"
              << "  --measured-steps N     Default: 10\n"
              << "  --input-seed N         Default: 1337\n"
              << "  --raw-output-prefix P  Optional prefix for binary output dump\n"
              << "  --output FILE          Default: aix_p2p_steps.csv\n";
    std::exit(1);
}

int parse_int(const char* value, const char* name)
{
    try {
        const long parsed = std::stol(value);
        if (parsed < 0 || parsed > std::numeric_limits<int>::max()) {
            throw std::out_of_range(name);
        }
        return static_cast<int>(parsed);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("Invalid value for ") + name + ": " + value);
    }
}

int64_t parse_int64(const char* value, const char* name)
{
    try {
        const long long parsed = std::stoll(value);
        if (parsed < 1) {
            throw std::out_of_range(name);
        }
        return static_cast<int64_t>(parsed);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("Invalid value for ") + name + ": " + value);
    }
}

std::vector<int64_t> parse_shape(const char* value, const char* name)
{
    std::vector<int64_t> shape;
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (!token.empty()) {
            shape.push_back(parse_int64(token.c_str(), name));
        }
    }
    if (shape.empty()) {
        throw std::runtime_error(std::string("Empty shape provided for ") + name);
    }
    return shape;
}

uint64_t splitmix64(uint64_t state)
{
    uint64_t z = (state + 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

void fill_deterministic_input(float* data, int64_t total_elements, int rank, int logical_step, int seed)
{
    for (int64_t element = 0; element < total_elements; ++element) {
        uint64_t key = static_cast<uint64_t>(seed);
        key ^= (static_cast<uint64_t>(rank + 1) * 0x100000001b3ULL);
        key ^= (static_cast<uint64_t>(logical_step + 1) * 0x9e3779b9ULL);
        key ^= (static_cast<uint64_t>(element) * 0x85ebca6bULL);
        const uint64_t hash = splitmix64(key);
        // Map top 23 bits to float in [-1.0, 1.0)
        const uint32_t mantissa = static_cast<uint32_t>(hash >> 41);
        const float unit = static_cast<float>(mantissa) / 8388608.0F; // 2^23
        data[element] = unit - 1.0F;
    }
}

double calibrate_clock_offset(MPI_Comm communicator, int rank, int size)
{
    constexpr int samples = 64;
    if (rank == 0) {
        for (int sample = 0; sample < samples; ++sample) {
            MPI_Barrier(communicator);
            double reference_time = MPI_Wtime();
            MPI_Bcast(&reference_time, 1, MPI_DOUBLE, 0, communicator);
        }
        return 0.0;
    }

    // A broadcast carries one root timestamp to every rank concurrently. The
    // largest observed offset is the sample with the smallest receive delay.
    double offset = -std::numeric_limits<double>::infinity();
    for (int sample = 0; sample < samples; ++sample) {
        MPI_Barrier(communicator);
        double reference_time = 0.0;
        MPI_Bcast(&reference_time, 1, MPI_DOUBLE, 0, communicator);
        offset = std::max(offset, reference_time - MPI_Wtime());
    }
    return offset;
}

Options parse_options(int argc, char** argv)
{
    Options options;
    bool custom_input_shape = false;
    bool custom_output_shape = false;

    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        const auto value = [&]() -> const char* {
            if (++i >= argc) {
                throw std::runtime_error("Missing value for " + argument);
            }
            return argv[i];
        };
        if (argument == "--model") {
            options.model = value();
        } else if (argument == "--output") {
            options.output = value();
        } else if (argument == "--raw-output-prefix") {
            options.raw_output_prefix = value();
        } else if (argument == "--samples-per-rank") {
            options.samples_per_rank = parse_int64(value(), "--samples-per-rank");
        } else if (argument == "--batch-size") {
            options.batch_size = parse_int(value(), "--batch-size");
        } else if (argument == "--warmup-steps") {
            options.warmup_steps = parse_int(value(), "--warmup-steps");
        } else if (argument == "--measured-steps") {
            options.measured_steps = parse_int(value(), "--measured-steps");
        } else if (argument == "--input-seed") {
            options.input_seed = parse_int(value(), "--input-seed");
        } else if (argument == "--input-shape") {
            options.input_shape = parse_shape(value(), "--input-shape");
            custom_input_shape = true;
        } else if (argument == "--output-shape") {
            options.output_shape = parse_shape(value(), "--output-shape");
            custom_output_shape = true;
        } else if (argument == "--help" || argument == "-h") {
            usage(argv[0]);
        } else {
            throw std::runtime_error("Unknown option: " + argument);
        }
    }
    if (options.model.empty()) {
        throw std::runtime_error("--model is required");
    }
    if (options.batch_size < 1 || options.measured_steps < 1) {
        throw std::runtime_error("Batch size and measured steps must be positive.");
    }

    if (!custom_input_shape) {
        options.input_shape = {options.samples_per_rank, 18};
    } else {
        options.input_shape[0] = options.samples_per_rank;
    }

    if (!custom_output_shape) {
        options.output_shape = {options.samples_per_rank};
    } else {
        options.output_shape[0] = options.samples_per_rank;
    }

    return options;
}

void check_mpi(int error, const char* operation)
{
    if (error == MPI_SUCCESS) {
        return;
    }
    char message[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    MPI_Error_string(error, message, &length);
    throw std::runtime_error(std::string(operation) + ": " + std::string(message, length));
}
} // namespace

int main(int argc, char** argv)
{
    int mpi_thread_level = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_thread_level);
    if (mpi_thread_level < MPI_THREAD_MULTIPLE) {
        std::cerr << "MPI implementation does not provide MPI_THREAD_MULTIPLE; "
                  << "the P2P controller will use synchronous inference." << '\n';
    }
    int rank = -1;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try {
        const Options options = parse_options(argc, argv);
        const double clock_offset = calibrate_clock_offset(MPI_COMM_WORLD, rank, size);
        std::ostringstream clock_offset_stream;
        clock_offset_stream << std::setprecision(17) << clock_offset;
        const std::string clock_offset_text = clock_offset_stream.str();
        setenv("AIX_P2P_CLOCK_OFFSET_SECONDS", clock_offset_text.c_str(), 1);
        const char* timeline_directory = std::getenv("AIX_P2P_TIMELINE_DIR");
        if (rank == 0) {
            write_timeline_metadata(timeline_directory, options);
        }
        std::vector<TimelineMark> solver_timeline;
        const auto mark_solver_event = [&](int step, const char* event) {
            const std::string step_text = std::to_string(step);
            setenv("AIX_P2P_TIMELINE_STEP", step_text.c_str(), 1);
            if (timeline_directory) {
                solver_timeline.push_back({step, corrected_time(clock_offset), event});
            }
        };
        const auto elements_from_shape = [](const std::vector<int64_t>& shape) {
            return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
        };
        const int64_t input_elements = elements_from_shape(options.input_shape);
        const int64_t output_elements = elements_from_shape(options.output_shape);
        std::vector<float> input(static_cast<size_t>(input_elements));
        std::vector<float> output(static_cast<size_t>(output_elements), std::numeric_limits<float>::quiet_NaN());

        if (rank == 0) {
            std::cout << "AIX_P2P_CONFIG ranks=" << size
                      << " samples_per_rank=" << options.samples_per_rank
                      << " input_shape=";
            for (size_t d = 0; d < options.input_shape.size(); ++d) {
                std::cout << (d == 0 ? "[" : ",") << options.input_shape[d];
            }
            std::cout << "] output_shape=";
            for (size_t d = 0; d < options.output_shape.size(); ++d) {
                std::cout << (d == 0 ? "[" : ",") << options.output_shape[d];
            }
            std::cout << "]"
                      << " batch_size=" << options.batch_size
                      << " communication_mode="
                      << (std::getenv("AIX_COMMUNICATION_MODE") ? std::getenv("AIX_COMMUNICATION_MODE") : "collective")
                      << " clock_sync_samples=64"
                      << '\n';
        }

        AIxeleratorService<float> service(options.model, const_cast<std::vector<int64_t>&>(options.input_shape),
                                          input.data(), const_cast<std::vector<int64_t>&>(options.output_shape),
                                          output.data(), options.batch_size, MPI_COMM_WORLD);

        for (int step = 0; step < options.warmup_steps; ++step) {
            fill_deterministic_input(input.data(), input_elements, rank, step, options.input_seed);
            std::fill(output.begin(), output.end(), std::numeric_limits<float>::quiet_NaN());
            mark_solver_event(step, "solver_ml_step_start");
            service.inference();
            mark_solver_event(step, "solver_ml_step_end");
        }
        check_mpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier after warm-up");

        std::ofstream csv;
        if (rank == 0) {
            csv.open(options.output);
            if (!csv) {
                throw std::runtime_error("Could not open output file: " + options.output);
            }
            csv << "step,rank,local_step_ms,global_step_ms\n";
        }
        std::vector<double> rank_times(rank == 0 ? size : 0);

        MPI_File raw_output_file = MPI_FILE_NULL;
        if (!options.raw_output_prefix.empty()) {
            const std::string raw_bin_path = options.raw_output_prefix + ".f32";
            check_mpi(MPI_File_open(MPI_COMM_WORLD, raw_bin_path.c_str(),
                                   MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &raw_output_file),
                      "MPI_File_open raw output");
            if (rank == 0) {
                const std::string raw_json_path = options.raw_output_prefix + ".json";
                std::ofstream manifest(raw_json_path);
                manifest << "{\n"
                         << "  \"format\": \"aix-p2p-output-v1\",\n"
                         << "  \"dtype\": \"float32-le\",\n"
                         << "  \"shape\": [" << options.measured_steps << ", " << size;
                for (auto dim : options.output_shape) {
                    manifest << ", " << dim;
                }
                manifest << "],\n"
                         << "  \"warmup_steps\": " << options.warmup_steps << ",\n"
                         << "  \"input_seed\": " << options.input_seed << ",\n"
                         << "  \"model\": \"" << std::filesystem::path(options.model).filename().string() << "\",\n"
                         << "  \"communication_mode\": \""
                         << (std::getenv("AIX_COMMUNICATION_MODE") ? std::getenv("AIX_COMMUNICATION_MODE") : "collective") << "\"\n"
                         << "}\n";
            }
        }

#ifdef USE_SCOREP
        SCOREP_USER_REGION_DEFINE(step_region)
#endif

        bool all_steps_valid = true;
        for (int step = 0; step < options.measured_steps; ++step) {
            fill_deterministic_input(input.data(), input_elements, rank, options.warmup_steps + step, options.input_seed);
            std::fill(output.begin(), output.end(), std::numeric_limits<float>::quiet_NaN());
#ifdef USE_SCOREP
            SCOREP_USER_REGION_BEGIN(step_region, "aix_p2p_benchmark_step", SCOREP_USER_REGION_TYPE_COMMON)
#endif
            const auto start = std::chrono::steady_clock::now();
            mark_solver_event(options.warmup_steps + step, "solver_ml_step_start");
            service.inference();
            mark_solver_event(options.warmup_steps + step, "solver_ml_step_end");
            const double local_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
#ifdef USE_SCOREP
            SCOREP_USER_REGION_END(step_region)
#endif

            const bool step_valid = std::all_of(output.begin(), output.end(), [](float value) {
                return std::isfinite(value);
            });
            if (!step_valid) {
                all_steps_valid = false;
            }

            if (raw_output_file != MPI_FILE_NULL) {
                const MPI_Offset step_bytes = static_cast<MPI_Offset>(size) * output_elements * sizeof(float);
                const MPI_Offset offset = static_cast<MPI_Offset>(step) * step_bytes +
                    static_cast<MPI_Offset>(rank) * output_elements * sizeof(float);
                MPI_Status status;
                check_mpi(MPI_File_write_at_all(raw_output_file, offset, output.data(),
                                                static_cast<int>(output_elements), MPI_FLOAT, &status),
                          "MPI_File_write_at_all raw output");
            }

            double global_ms = 0.0;
            check_mpi(MPI_Reduce(&local_ms, &global_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD), "MPI_Reduce");
            check_mpi(MPI_Gather(&local_ms, 1, MPI_DOUBLE, rank_times.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD), "MPI_Gather");
            if (rank == 0) {
                for (int source_rank = 0; source_rank < size; ++source_rank) {
                    csv << step << ',' << source_rank << ',' << std::setprecision(12) << rank_times[source_rank] << ',' << global_ms << '\n';
                }
            }
            check_mpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier after CSV collection");
        }

        if (raw_output_file != MPI_FILE_NULL) {
            check_mpi(MPI_File_close(&raw_output_file), "MPI_File_close raw output");
        }

        int all_valid = 0;
        const int local_valid = all_steps_valid ? 1 : 0;
        check_mpi(MPI_Allreduce(&local_valid, &all_valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD), "MPI_Allreduce");

        const char* measure_mem_env = std::getenv("AIX_MEASURE_MEMORY");
        const bool measure_mem = (measure_mem_env == nullptr || std::string(measure_mem_env) != "0");

        if (rank == 0) {
            std::cout << "AIX_P2P_RESULT output_valid=" << all_valid << " csv=" << options.output << '\n';
        }

        if (measure_mem) {
            const double local_rss_mb = get_local_peak_rss_mb();
            double max_cpu_rss_mb = 0.0;
            double sum_cpu_rss_mb = 0.0;
            check_mpi(MPI_Reduce(&local_rss_mb, &max_cpu_rss_mb, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD), "MPI_Reduce max RSS");
            check_mpi(MPI_Reduce(&local_rss_mb, &sum_cpu_rss_mb, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD), "MPI_Reduce sum RSS");

            double local_gpu_vram_mb = (rank == size - 1) ? get_gpu_vram_used_mb() : 0.0;
            double gpu_vram_used_mb = 0.0;
            check_mpi(MPI_Reduce(&local_gpu_vram_mb, &gpu_vram_used_mb, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD), "MPI_Reduce GPU VRAM");

            if (rank == 0) {
                std::cout << "AIX_P2P_MEMORY max_cpu_rss_mb=" << std::fixed << std::setprecision(2) << max_cpu_rss_mb 
                          << " sum_cpu_rss_mb=" << sum_cpu_rss_mb 
                          << " gpu_vram_used_mb=" << gpu_vram_used_mb << '\n';
            }
        }
        write_solver_timeline(timeline_directory, rank, solver_timeline);
        check_mpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier before finalize");
        MPI_Finalize();
        return all_valid ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "AIX_P2P_ERROR rank=" << rank << " " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
