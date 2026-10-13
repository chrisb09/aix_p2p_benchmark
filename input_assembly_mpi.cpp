#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kCreditTag = 17001;
constexpr int kInputTag = 17002;
constexpr int kInputAckTag = 17004;
constexpr int kDefaultPayloadBytes = 6'220'800;  // 320 x 270 x 18 FP32 values per 96c4g rank.
constexpr int kMetricsPerIteration = 10;

enum class Mode { Collective, P2PCredit1, P2PFullCredits };

struct Options {
    Mode mode = Mode::Collective;
    int group_size = 24;
    int payload_bytes = kDefaultPayloadBytes;
    int warmups = 20;
    int iterations = 100;
    std::string output = "input_assembly.csv";
};

const char* mode_name(Mode mode) {
    switch (mode) {
        case Mode::Collective: return "collective_gatherv";
        case Mode::P2PCredit1: return "p2p_credit1";
        case Mode::P2PFullCredits: return "p2p_fullcredits";
    }
    return "unknown";
}

[[noreturn]] void usage_and_exit(const char* message = nullptr) {
    if (message != nullptr) {
        std::cerr << "Error: " << message << "\n\n";
    }
    std::cerr << "Usage: aix_input_assembly_mpi --mode collective|p2p-credit1|p2p-fullcredits "
                 "[--group-size 24] [--payload-bytes 6220800] [--warmups 20] "
                 "[--iterations 100] [--output FILE]\n";
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
            if (value == "collective") {
                options.mode = Mode::Collective;
            } else if (value == "p2p-credit1") {
                options.mode = Mode::P2PCredit1;
            } else if (value == "p2p-fullcredits") {
                options.mode = Mode::P2PFullCredits;
            } else {
                throw std::runtime_error("--mode must be collective, p2p-credit1, or p2p-fullcredits");
            }
        } else if (arg == "--group-size") {
            options.group_size = parse_positive_int(value, "--group-size");
        } else if (arg == "--payload-bytes") {
            options.payload_bytes = parse_positive_int(value, "--payload-bytes");
        } else if (arg == "--warmups") {
            options.warmups = parse_positive_int(value, "--warmups");
        } else if (arg == "--iterations") {
            options.iterations = parse_positive_int(value, "--iterations");
        } else if (arg == "--output") {
            options.output = value;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return options;
}

void check_mpi(int result, const char* operation, MPI_Comm comm) {
    if (result == MPI_SUCCESS) {
        return;
    }
    char error[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(result, error, &length);
    std::cerr << operation << " failed: " << std::string(error, length) << "\n";
    MPI_Abort(comm, result);
}

void write_results(const Options& options, int world_size, int provided_thread_level,
                   const std::vector<double>& all_rows) {
    std::ofstream stream(options.output);
    if (!stream) {
        throw std::runtime_error("cannot open output file: " + options.output);
    }
    const int groups = world_size / options.group_size;
    stream << "# mode=" << mode_name(options.mode)
           << ",world_size=" << world_size
           << ",group_size=" << options.group_size
           << ",workgroups=" << groups
           << ",payload_bytes_per_rank=" << options.payload_bytes
           << ",mpi_thread_level_provided=" << provided_thread_level
           << ",warmups=" << options.warmups
           << ",iterations=" << options.iterations << "\n";
    stream << "mode,scenario_world_ranks,group_size,workgroups,iteration,workgroup,"
              "assembly_ms,aggregate_input_gib_s,credit_wait_mean_ms,credit_wait_max_ms,"
              "send_mean_ms,send_max_ms,ack_wait_mean_ms,ack_wait_max_ms\n";
    stream << std::fixed << std::setprecision(6);
    for (int rank = 0; rank < world_size; ++rank) {
        for (int iteration = 0; iteration < options.iterations; ++iteration) {
            const size_t base = (static_cast<size_t>(rank) * options.iterations + iteration) * kMetricsPerIteration;
            if (std::isnan(all_rows[base])) {
                continue;
            }
            stream << mode_name(options.mode) << ',' << world_size << ',' << options.group_size << ',' << groups << ','
                   << iteration << ',' << static_cast<int>(all_rows[base + 1]) << ','
                   << all_rows[base + 2] << ',' << all_rows[base + 3] << ','
                   << all_rows[base + 4] << ',' << all_rows[base + 5] << ','
                   << all_rows[base + 6] << ',' << all_rows[base + 7] << ','
                   << all_rows[base + 8] << ',' << all_rows[base + 9] << '\n';
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    int provided_thread_level = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided_thread_level);

    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    try {
        const Options options = parse_options(argc, argv);
        if (world_size % options.group_size != 0) {
            throw std::runtime_error("world size must be divisible by --group-size");
        }
        if (options.payload_bytes > std::numeric_limits<int>::max() / options.group_size) {
            throw std::runtime_error("payload is too large for MPI int counts");
        }

        const int workgroup = world_rank / options.group_size;
        MPI_Comm group_comm = MPI_COMM_NULL;
        check_mpi(MPI_Comm_split(MPI_COMM_WORLD, workgroup, world_rank, &group_comm), "MPI_Comm_split", MPI_COMM_WORLD);
        int group_rank = 0;
        MPI_Comm_rank(group_comm, &group_rank);

        std::vector<unsigned char> send_buffer(options.payload_bytes, static_cast<unsigned char>(world_rank));
        std::vector<unsigned char> receive_buffer;
        if (group_rank == 0) {
            receive_buffer.resize(static_cast<size_t>(options.payload_bytes) * options.group_size);
        }

        std::vector<int> counts(options.group_size, options.payload_bytes);
        std::vector<int> displacements(options.group_size, 0);
        for (int rank = 0; rank < options.group_size; ++rank) {
            displacements[rank] = rank * options.payload_bytes;
        }

        std::vector<double> local_rows(static_cast<size_t>(options.iterations) * kMetricsPerIteration,
                                       std::numeric_limits<double>::quiet_NaN());
        const int total_iterations = options.warmups + options.iterations;

        for (int iteration = 0; iteration < total_iterations; ++iteration) {
            std::vector<MPI_Request> input_requests(options.group_size, MPI_REQUEST_NULL);
            std::vector<MPI_Request> credit_requests(options.group_size, MPI_REQUEST_NULL);
            std::vector<MPI_Request> ack_requests(options.group_size, MPI_REQUEST_NULL);
            std::vector<int> credit_values(options.group_size, 1);
            std::vector<int> ack_values(options.group_size, 1);

            // AIx posts every receive before issuing credits. All ranks in this one-node
            // fixture are in tier 0, matching the production locality-tier algorithm.
            if (options.mode != Mode::Collective && group_rank == 0) {
                for (int rank = 1; rank < options.group_size; ++rank) {
                    check_mpi(MPI_Irecv(receive_buffer.data() + displacements[rank], options.payload_bytes, MPI_BYTE,
                                        rank, kInputTag, group_comm, &input_requests[rank]),
                              "MPI_Irecv", group_comm);
                }
            }

            check_mpi(MPI_Barrier(group_comm), "MPI_Barrier", group_comm);

            double assembly_ms = 0.0;
            double worker_credit_wait_ms = 0.0;
            double worker_send_ms = 0.0;
            double worker_ack_wait_ms = 0.0;

            if (options.mode == Mode::Collective) {
                if (group_rank == 0) {
                    const double begin = MPI_Wtime();
                    check_mpi(MPI_Gatherv(send_buffer.data(), options.payload_bytes, MPI_BYTE,
                                          receive_buffer.data(), counts.data(), displacements.data(), MPI_BYTE,
                                          0, group_comm), "MPI_Gatherv", group_comm);
                    assembly_ms = (MPI_Wtime() - begin) * 1.0e3;
                } else {
                    check_mpi(MPI_Gatherv(send_buffer.data(), options.payload_bytes, MPI_BYTE,
                                          nullptr, nullptr, nullptr, MPI_BYTE, 0, group_comm),
                              "MPI_Gatherv", group_comm);
                }
            } else if (group_rank == 0) {
                const int initial_credits = options.mode == Mode::P2PCredit1 ? 1 : options.group_size - 1;
                int next_rank = 1;
                const auto issue_grants = [&](int requested) {
                    const int grant_count = std::min(requested, options.group_size - next_rank);
                    for (int grant = 0; grant < grant_count; ++grant) {
                        const int destination = next_rank++;
                        check_mpi(MPI_Isend(&credit_values[destination], 1, MPI_INT, destination, kCreditTag,
                                            group_comm, &credit_requests[destination]), "MPI_Isend credit", group_comm);
                    }
                };

                const double begin = MPI_Wtime();
                issue_grants(initial_credits);
                int received = 0;
                std::vector<int> completed_indices(options.group_size);
                std::vector<MPI_Status> completed_statuses(options.group_size);
                while (received < options.group_size - 1) {
                    int completed = 0;
                    check_mpi(MPI_Testsome(options.group_size, input_requests.data(), &completed,
                                           completed_indices.data(), completed_statuses.data()),
                              "MPI_Testsome", group_comm);
                    if (completed == MPI_UNDEFINED || completed == 0) {
                        continue;
                    }
                    for (int index = 0; index < completed; ++index) {
                        const int source = completed_indices[index];
                        ++received;
                        check_mpi(MPI_Isend(&ack_values[source], 1, MPI_INT, source, kInputAckTag,
                                            group_comm, &ack_requests[source]), "MPI_Isend input ACK", group_comm);
                        if (options.mode == Mode::P2PCredit1) {
                            issue_grants(2);
                        }
                    }
                }
                assembly_ms = (MPI_Wtime() - begin) * 1.0e3;
                check_mpi(MPI_Waitall(options.group_size, credit_requests.data(), MPI_STATUSES_IGNORE),
                          "MPI_Waitall credits", group_comm);
                check_mpi(MPI_Waitall(options.group_size, ack_requests.data(), MPI_STATUSES_IGNORE),
                          "MPI_Waitall ACKs", group_comm);
            } else {
                const double credit_begin = MPI_Wtime();
                int credit = 0;
                check_mpi(MPI_Recv(&credit, 1, MPI_INT, 0, kCreditTag, group_comm, MPI_STATUS_IGNORE),
                          "MPI_Recv credit", group_comm);
                const double send_begin = MPI_Wtime();
                check_mpi(MPI_Send(send_buffer.data(), options.payload_bytes, MPI_BYTE, 0, kInputTag, group_comm),
                          "MPI_Send input", group_comm);
                const double ack_begin = MPI_Wtime();
                int ack = 0;
                check_mpi(MPI_Recv(&ack, 1, MPI_INT, 0, kInputAckTag, group_comm, MPI_STATUS_IGNORE),
                          "MPI_Recv input ACK", group_comm);
                const double end = MPI_Wtime();
                worker_credit_wait_ms = (send_begin - credit_begin) * 1.0e3;
                worker_send_ms = (ack_begin - send_begin) * 1.0e3;
                worker_ack_wait_ms = (end - ack_begin) * 1.0e3;
            }

            std::array<double, 3> worker_sums = {worker_credit_wait_ms, worker_send_ms, worker_ack_wait_ms};
            std::array<double, 3> worker_maxes = worker_sums;
            std::array<double, 3> worker_sum_result = {0.0, 0.0, 0.0};
            std::array<double, 3> worker_max_result = {0.0, 0.0, 0.0};
            check_mpi(MPI_Reduce(worker_sums.data(), worker_sum_result.data(), 3, MPI_DOUBLE, MPI_SUM, 0, group_comm),
                      "MPI_Reduce worker sums", group_comm);
            check_mpi(MPI_Reduce(worker_maxes.data(), worker_max_result.data(), 3, MPI_DOUBLE, MPI_MAX, 0, group_comm),
                      "MPI_Reduce worker maxima", group_comm);

            if (iteration >= options.warmups && group_rank == 0) {
                const int measured_iteration = iteration - options.warmups;
                const size_t base = static_cast<size_t>(measured_iteration) * kMetricsPerIteration;
                const double workers = static_cast<double>(options.group_size - 1);
                const double aggregate_gib_s = (workers * options.payload_bytes / (1024.0 * 1024.0 * 1024.0)) /
                    (assembly_ms / 1.0e3);
                local_rows[base] = static_cast<double>(measured_iteration);
                local_rows[base + 1] = static_cast<double>(workgroup);
                local_rows[base + 2] = assembly_ms;
                local_rows[base + 3] = aggregate_gib_s;
                local_rows[base + 4] = worker_sum_result[0] / workers;
                local_rows[base + 5] = worker_max_result[0];
                local_rows[base + 6] = worker_sum_result[1] / workers;
                local_rows[base + 7] = worker_max_result[1];
                local_rows[base + 8] = worker_sum_result[2] / workers;
                local_rows[base + 9] = worker_max_result[2];
            }
        }

        std::vector<double> all_rows;
        if (world_rank == 0) {
            all_rows.resize(static_cast<size_t>(world_size) * options.iterations * kMetricsPerIteration);
        }
        check_mpi(MPI_Gather(local_rows.data(), static_cast<int>(local_rows.size()), MPI_DOUBLE,
                             all_rows.data(), static_cast<int>(local_rows.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD),
                  "MPI_Gather metrics", MPI_COMM_WORLD);
        if (world_rank == 0) {
            write_results(options, world_size, provided_thread_level, all_rows);
            std::cout << "Wrote " << options.output << " for " << mode_name(options.mode)
                      << " with " << world_size / options.group_size << " concurrent workgroups.\n";
        }

        MPI_Comm_free(&group_comm);
        MPI_Finalize();
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        if (world_rank == 0) {
            std::cerr << "Fatal error: " << error.what() << '\n';
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    return EXIT_FAILURE;
}
