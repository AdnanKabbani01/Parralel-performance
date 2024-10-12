#include <mpi.h>
#include <stdio.h>
#include <math.h>

// Function to evaluate the curve (y = f(x))
double f(double x) {
    return x * x; // Example: y = x^2
}

// Function to compute the area of a trapezoid and count computation steps
double trapezoid_area(double a, double b, double d, unsigned long long *steps) { 
    double area = 0.0;
    unsigned long long local_steps = 0;

    for (double x = a; x < b; x += d) {
        area += f(x) + f(x + d);
        local_steps += 2; 
    }

    *steps = local_steps;
    return area * d / 2.0;
}

int main(int argc, char** argv) {
    int rank, size;
    double a = 0.0, b = 1.0;  // Limits of integration
    unsigned long long n;
    double start_time, end_time, computation_time;
    double local_area, total_area;
    unsigned long long local_steps, total_steps;

    // Number of iterations for averaging
    int iterations = 100;

    MPI_Init(&argc, &argv); // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get number of processes

    if (rank == 0) {
        // Get the number of intervals from the user
        printf("Enter the number of intervals: ");
        fflush(stdout);
        scanf("%llu", &n);
    }

    // Broadcast the number of intervals to all processes
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    // Calculate the interval size for each processes
    double d = (b - a) / n; // delta
    double region = (b - a) / size;

    // Calculate local bounds for each process
    double start = a + rank * region;
    double end = start + region;

    // Initialize computation time and local steps
    computation_time = 0.0;
    local_steps = 0;
    local_area = 0.0;

    // Perform multiple iterations to average the execution time
    for (int i = 0; i < iterations; i++) {
        // Synchronize before starting computation
        MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();

        // Each process calculates the area of its subinterval
        double area = trapezoid_area(start, end, d, &local_steps);
        local_area += area;

        // Reduce all local areas to the total area on the root process
        MPI_Reduce(&local_area, &total_area, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        end_time = MPI_Wtime();
        computation_time += (end_time - start_time);
    }

    // Calculate average computation time
    computation_time /= iterations;

    //get the maximum computation time across all processes
    double max_time;
    MPI_Reduce(&computation_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    //get the total computation steps
    MPI_Reduce(&local_steps, &total_steps, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("The total area under the curve is: %.6f\n", total_area);
        printf("Average computation time over %d iterations: %.6f seconds\n", iterations, max_time);
        printf("Total computation steps: %llu\n", total_steps);
    }

    MPI_Finalize(); // Finalize MPI
    return 0;
}
