import numpy as np
import sys
import time
import subprocess
import os
import csv
from statistics import mean, stdev
from math import sqrt


def generate_symmetric_graph(n, seed=42):
    np.random.seed(seed)
    upper = np.random.randint(1, 100, size=(n, n)).astype(np.float64)
    D = np.triu(upper, 1)
    D += D.T
    return D


def floyd_warshall_sequential(D):
    n = D.shape[0]
    dist = D.copy()
    start_time = time.time()
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, j] > dist[i, k] + dist[k, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
    total_time = time.time() - start_time
    return dist, total_time


def compare_results(sequential_result, parallel_result_file, num_processes):
    if os.path.exists(parallel_result_file):
        parallel_dist = np.load(parallel_result_file)
        if np.allclose(sequential_result, parallel_dist):
            print(f"numprocs {num_processes} result ok.")
        else:
            print(f"numprocs {num_processes} result DIFFER.")
    else:
        print(f"numprocs {parallel_result_file} file not found.")


def run_parallel_version(n, output_dir, num_processes, sequential_result, num_repeats):
    times = []
    for run in range(num_repeats):
        command = ["mpiexec", "-np", str(num_processes),
                   "python", "parallel_script.py", str(n), output_dir]

        print(f"\nRun {run+1}/{num_repeats}: {' '.join(command)}")
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, check=True)
            print(result.stdout)
            for line in result.stdout.splitlines():
                if "Time with" in line and "processes" in line:
                    try:
                        time_str = line.split(":")[-1].strip().split(" ")[0]
                        times.append(float(time_str))
                    except ValueError:
                        print(f"Failed to parse time from line: {line}")
        except subprocess.CalledProcessError as e:
            print("Subprocess failed:")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")

    parallel_result_file = os.path.join(
        output_dir, f"parallel_result_{num_processes}.npy")
    compare_results(sequential_result, parallel_result_file, num_processes)

    if times:
        mean_time = mean(times)
        stderr = stdev(times) / sqrt(num_repeats) if num_repeats > 1 else 0.0
        return mean_time, stderr
    return None, None


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python main.py <number_of_vertices> <max_number_of_processes> <num_repeats>")
        sys.exit(1)

    n = int(sys.argv[1])
    max_processes = int(sys.argv[2])
    num_repeats = int(sys.argv[3])

    output_dir = "floyd_warshall_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"All output files will be saved to: {output_dir}")

    print(f"Generating graph with {n} vertices...")
    graph = generate_symmetric_graph(n)
    np.save(os.path.join(output_dir, "input_graph.npy"), graph)

    print("\nRunning sequential Floyd-Warshall algorithm...")
    seq_dist, seq_time = floyd_warshall_sequential(graph)
    np.save(os.path.join(output_dir, "sequential_result.npy"), seq_dist)
    sequential_result = np.load(os.path.join(
        output_dir, "sequential_result.npy"))
    print(f"Sequential version time: {seq_time:.4f} seconds")

    parallel_stats = {}

    print("\nRunning parallel Floyd-Warshall algorithm with different number of processes...")
    for num_processes in range(2, max_processes + 1):
        mean_time, stderr = run_parallel_version(
            n, output_dir, num_processes, sequential_result, num_repeats)
        if mean_time is not None:
            parallel_stats[num_processes] = (mean_time, stderr)

    print("\nExecution time summary:")
    print(f"Sequential version: {seq_time:.4f} seconds")
    for num_procs, (mean_time, stderr) in parallel_stats.items():
        print(
            f"Parallel ({num_procs} proc): mean = {mean_time:.4f}s, stderr = {stderr:.4f}s")

    csv_dir = "csv_result"
    os.makedirs(csv_dir, exist_ok=True)

    csv_filename = os.path.join(
        csv_dir, f"execution_times_n{n}_maxp{max_processes}.csv")
    with open(csv_filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["numprocs", "mean_time (s)", "stderr (s)"])

        # Add sequential as 1 proc
        writer.writerow([1, f"{seq_time:.4f}", "0.0000"])

        # Add parallel runs
        for num_procs in sorted(parallel_stats.keys()):
            mean_time, stderr = parallel_stats[num_procs]
            writer.writerow([num_procs, f"{mean_time:.4f}", f"{stderr:.4f}"])
