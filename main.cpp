#include "aixeleratorService/aixeleratorService.h"

#include <mpi.h>

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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Options {
    std::string model;
    std::string output = "aix_p2p_steps.csv";
    int64_t samples_per_rank = 129600;
    int batch_size = 4000000;
    int warmup_steps = 3;
    int measured_steps = 10;
    int input_width = 18;
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
              << "  --samples-per-rank N  Default: 129600\n"
              << "  --batch-size N        Default: 4000000\n"
              << "  --warmup-steps N      Default: 3\n"
              << "  --measured-steps N    Default: 10\n"
              << "  --output FILE         Default: aix_p2p_steps.csv\n";
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
        } else if (argument == "--samples-per-rank") {
            options.samples_per_rank = parse_int64(value(), "--samples-per-rank");
        } else if (argument == "--batch-size") {
            options.batch_size = parse_int(value(), "--batch-size");
        } else if (argument == "--warmup-steps") {
            options.warmup_steps = parse_int(value(), "--warmup-steps");
        } else if (argument == "--measured-steps") {
            options.measured_steps = parse_int(value(), "--measured-steps");
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
        const std::vector<int64_t> input_shape = {options.samples_per_rank, options.input_width};
        const std::vector<int64_t> output_shape = {options.samples_per_rank};
        std::vector<float> input(static_cast<size_t>(options.samples_per_rank * options.input_width));
        std::vector<float> output(static_cast<size_t>(options.samples_per_rank), -13.37F);
        for (size_t element = 0; element < input.size(); ++element) {
            input[element] = static_cast<float>(rank) + static_cast<float>(element % options.input_width) * 0.01F;
        }

        if (rank == 0) {
            std::cout << "AIX_P2P_CONFIG ranks=" << size
                      << " samples_per_rank=" << options.samples_per_rank
                      << " input_shape=[N," << options.input_width << "]"
                      << " output_shape=[N]"
                      << " batch_size=" << options.batch_size
                      << " communication_mode="
                      << (std::getenv("AIX_COMMUNICATION_MODE") ? std::getenv("AIX_COMMUNICATION_MODE") : "collective")
                      << " clock_sync_samples=64"
                      << '\n';
        }

        AIxeleratorService<float> service(options.model, input_shape, input.data(), output_shape, output.data(),
                                          options.batch_size, MPI_COMM_WORLD);

        for (int step = 0; step < options.warmup_steps; ++step) {
            mark_solver_event(step, "solver_ml_step_start");
            service.inference();
            mark_solver_event(step, "solver_ml_step_end");
        }
        // Align the first measured call after warm-up without imposing a barrier
        // on subsequent steps, whose rank-arrival spread remains observable.
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

#ifdef USE_SCOREP
        SCOREP_USER_REGION_DEFINE(step_region)
#endif

        for (int step = 0; step < options.measured_steps; ++step) {
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

            double global_ms = 0.0;
            check_mpi(MPI_Reduce(&local_ms, &global_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD), "MPI_Reduce");
            check_mpi(MPI_Gather(&local_ms, 1, MPI_DOUBLE, rank_times.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD), "MPI_Gather");
            if (rank == 0) {
                for (int source_rank = 0; source_rank < size; ++source_rank) {
                    csv << step << ',' << source_rank << ',' << std::setprecision(12) << rank_times[source_rank] << ',' << global_ms << '\n';
                }
            }
            // The CSV collectives can return to non-root ranks before the root.
            // Keep that reporting skew out of the next synthetic ML-step timeline.
            if (step + 1 < options.measured_steps) {
                check_mpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier after CSV collection");
            }
        }

        const bool locally_valid = std::all_of(output.begin(), output.end(), [](float value) {
            return std::isfinite(value) && value != -13.37F;
        });
        int all_valid = 0;
        const int local_valid = locally_valid ? 1 : 0;
        check_mpi(MPI_Allreduce(&local_valid, &all_valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD), "MPI_Allreduce");
        if (rank == 0) {
            std::cout << "AIX_P2P_RESULT output_valid=" << all_valid << " csv=" << options.output << '\n';
        }
        write_solver_timeline(timeline_directory, rank, solver_timeline);
        MPI_Finalize();
        return all_valid ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "AIX_P2P_ERROR rank=" << rank << " " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
