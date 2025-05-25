# parallel_script.py
import numpy as np
import sys
import time
from mpi4py import MPI
import os

def floyd_warshall_parallel(n, output_dir):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        D = np.load(os.path.join(output_dir, "input_graph.npy"))
    else:
        D = np.empty((n, n), dtype=np.float64)

    comm.Bcast(D, root=0)

    rows_per_proc = n // size
    extra = n % size
    start_row = rank * rows_per_proc + min(rank, extra)
    end_row = start_row + rows_per_proc + (1 if rank < extra else 0)

    local_D = D[start_row:end_row, :].copy()

    comm.Barrier()
    start_time = time.time()

    for k in range(n):
        owner_rank = -1
        current_row_base_index = 0
        for r in range(size):
            r_rows_count = rows_per_proc + (1 if r < extra else 0)
            if current_row_base_index <= k < current_row_base_index + r_rows_count:
                owner_rank = r
                break
            current_row_base_index += r_rows_count

        row_k = np.empty(n, dtype=np.float64)

        if rank == owner_rank:
            local_k_index = k - start_row
            row_k = local_D[local_k_index, :].copy()

        comm.Bcast(row_k, root=owner_rank)

        for i in range(end_row - start_row):
            for j in range(n):
                if local_D[i, j] > local_D[i, k] + row_k[j]:
                    local_D[i, j] = local_D[i, k] + row_k[j]

    comm.Barrier()
    total_time = time.time() - start_time

    counts = [(rows_per_proc + (1 if i < extra else 0)) * n for i in range(size)]
    displs = [sum(counts[:i]) for i in range(size)]

    if rank == 0:
        final_D = np.empty((n, n), dtype=np.float64)
    else:
        final_D = None

    comm.Gatherv(local_D.flatten(), (final_D, counts, displs, MPI.DOUBLE), root=0)

    if rank == 0:
        np.save(os.path.join(output_dir, f"parallel_result_{size}.npy"), final_D)
        print(f"Time with {size} processes: {total_time:.4f} seconds") # Translated
        return total_time
    return None

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python parallel_script.py <number_of_vertices> <output_directory>") # Translated
        sys.exit(1)
    n = int(sys.argv[1])
    output_dir = sys.argv[2]
    floyd_warshall_parallel(n, output_dir)