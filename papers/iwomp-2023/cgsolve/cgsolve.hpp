//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <generate_matrix.hpp>

struct cgsolve {
  int N, max_iter;
  double tolerance;
  Kokkos::View<double *> y, x;
  CrsMatrix<Kokkos::DefaultExecutionSpace::memory_space> A;

  cgsolve(int N_, int max_iter_in, double tolerance_in)
      : N(N_), max_iter(max_iter_in), tolerance(tolerance_in) {
    CrsMatrix<Kokkos::HostSpace> h_A = Impl::generate_miniFE_matrix(N);
    Kokkos::View<double *, Kokkos::HostSpace> h_x =
        Impl::generate_miniFE_vector(N);

    Kokkos::View<int64_t *> row_ptr("row_ptr", h_A.row_ptr.extent(0));
    Kokkos::View<int64_t *> col_idx("col_idx", h_A.col_idx.extent(0));
    Kokkos::View<double *> values("values", h_A.values.extent(0));
    A = CrsMatrix<Kokkos::DefaultExecutionSpace::memory_space>(
        row_ptr, col_idx, values, h_A.num_cols());
    x = Kokkos::View<double *>("X", h_x.extent(0));
    y = Kokkos::View<double *>("Y", h_x.extent(0));

    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(A.row_ptr, h_A.row_ptr);
    Kokkos::deep_copy(A.col_idx, h_A.col_idx);
    Kokkos::deep_copy(A.values, h_A.values);
  }

  template <class YType, class AType, class XType>
  double spmv(YType y, AType A, XType x) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_SYCL)
    int rows_per_team = 16;
    int team_size     = 16;
    int vector_size   = 8;
#elif defined(KOKKOS_ENABLE_OPENMPTARGET)
#if defined(KOKKOS_ARCH_INTEL_GPU)
    int rows_per_team = 16;
    int team_size     = 16;
    int vector_size   = 1;
#else
    int rows_per_team = 32;
    int team_size     = 32;
    int vector_size   = 1;
#endif
#else
    int rows_per_team = 512;
    int team_size     = 1;
    int vector_size   = 1;
#endif
    int64_t nrows = y.extent(0);
    Kokkos::Timer timer;
    Kokkos::parallel_for(
        "SPMV",
        Kokkos::TeamPolicy<>((nrows + rows_per_team - 1) / rows_per_team,
                             team_size, vector_size),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
          const int64_t first_row = team.league_rank() * rows_per_team;
          const int64_t last_row  = first_row + rows_per_team < nrows
                                        ? first_row + rows_per_team
                                        : nrows;
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, first_row, last_row),
              [&](const int64_t row) {
                const int64_t row_start  = A.row_ptr(row);
                const int64_t row_length = A.row_ptr(row + 1) - row_start;

                double y_row;
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(team, row_length),
                    [=](const int64_t i, double &sum) {
                      sum +=
                          A.values(i + row_start) * x(A.col_idx(i + row_start));
                    },
                    y_row);
                y(row) = y_row;
              });
        });

    Kokkos::fence();
    double time = timer.seconds();

    return time;
  }

#if defined(KOKKOS_ENABLE_OPENMPTARGET)
  template <class YType, class AType, class XType>
  double spmv_ompt(YType y, AType A, XType x) {
    int rows_per_team = 32;
    int team_size     = 32;
    int64_t nrows     = y.extent(0);

    auto row_ptr = A.row_ptr.data();
    auto values  = A.values.data();
    auto col_idx = A.col_idx.data();
    auto xp      = x.data();
    auto yp      = y.data();

    int64_t n = (nrows + rows_per_team - 1) / rows_per_team;
    Kokkos::Timer timer;
#pragma omp target teams distribute is_device_ptr(row_ptr, values, col_idx, \
                                                      xp, yp)
    for (int64_t i = 0; i < n; ++i) {
#pragma omp parallel
      {
        const int64_t first_row = i * rows_per_team;
        const int64_t last_row  = first_row + rows_per_team < nrows
                                      ? first_row + rows_per_team
                                      : nrows;

#pragma omp for
        for (int64_t row = first_row; row < last_row; ++row) {
          const int64_t row_start  = row_ptr[row];
          const int64_t row_length = row_ptr[row + 1] - row_start;

          double y_row = 0.;
#pragma omp simd reduction(+ : y_row)
          for (int64_t i = 0; i < row_length; ++i) {
            y_row += values[i + row_start] * xp[col_idx[i + row_start]];
          }
          yp[row] = y_row;
        }
      }
    }
    double time = timer.seconds();

    return time;
  }
#endif

  template <class YType, class XType>
  double dot(YType y, XType x) {
    double result;
    Kokkos::parallel_reduce(
        "DOT", y.extent(0),
        KOKKOS_LAMBDA(const int64_t &i, double &lsum) { lsum += y(i) * x(i); },
        result);
    return result;
  }

#if defined(KOKKOS_ENABLE_OPENMPTARGET)
  template <class YType, class XType>
  double dot_ompt(YType y, XType x) {
    double result;
    int n   = y.extent(0);
    auto xp = x.data();
    auto yp = y.data();

#pragma omp barrier

#pragma omp target teams distribute parallel for is_device_ptr(xp, yp) \
    reduction(+ : result)
    for (int i = 0; i < n; ++i) {
      result += yp[i] * xp[i];
    }
    return result;
  }
#endif

  template <class ZType, class YType, class XType>
  void axpby(ZType z, double alpha, XType x, double beta, YType y) {
    int64_t n = z.extent(0);
    Kokkos::parallel_for(
        "AXPBY", n,
        KOKKOS_LAMBDA(const int &i) { z(i) = alpha * x(i) + beta * y(i); });
  }

#if defined(KOKKOS_ENABLE_OPENMPTARGET)
  template <class ZType, class YType, class XType>
  void axpby_ompt(ZType z, double alpha, XType x, double beta, YType y) {
    int64_t n = z.extent(0);
    auto xp   = x.data();
    auto yp   = y.data();
    auto zp   = z.data();

#pragma omp target teams distribute parallel for is_device_ptr(zp, yp, xp)
    for (int i = 0; i < n; ++i) {
      zp[i] = alpha * xp[i] + beta * yp[i];
    }
  }
#endif

  template <class VType>
  void print_vector(int label, VType v) {
    std::cout << "\n\nPRINT " << v.label() << std::endl << std::endl;

    int myRank = 0;
    Kokkos::parallel_for(
        v.extent(0), KOKKOS_LAMBDA(const int i) {
          printf("%i %i %i %lf\n", label, myRank, i, v(i));
        });
    Kokkos::fence();
    std::cout << "\n\nPRINT DONE " << v.label() << std::endl << std::endl;
  }

  template <class VType, class AType>
  int cg_solve_kk(VType y, AType A, VType b, int max_iter, double tolerance) {
    int myproc    = 0;
    int num_iters = 0;

    double normr     = 0;
    double rtrans    = 0;
    double oldrtrans = 0;

    double spmv_time = 0.;

    int64_t print_freq = max_iter / 10;
    if (print_freq > 50) print_freq = 50;
    if (print_freq < 1) print_freq = 1;
    VType x("x", b.extent(0));
    VType r("r", x.extent(0));
    VType p("r", x.extent(0));  // Needs to be global
    VType Ap("r", x.extent(0));
    double one  = 1.0;
    double zero = 0.0;
    axpby(p, one, x, zero, x);

    spmv_time += spmv(Ap, A, p);
    axpby(r, one, b, -one, Ap);

    rtrans = dot(r, r);

    normr = std::sqrt(rtrans);

    if (myproc == 0) {
      std::cout << "Initial Residual = " << normr << std::endl;
    }

    double brkdown_tol = std::numeric_limits<double>::epsilon();

    for (int64_t k = 1; k <= max_iter && normr > tolerance; ++k) {
      if (k == 1) {
        axpby(p, one, r, zero, r);
      } else {
        oldrtrans   = rtrans;
        rtrans      = dot(r, r);
        double beta = rtrans / oldrtrans;
        axpby(p, one, r, beta, p);
      }

      normr = std::sqrt(rtrans);

      double alpha    = 0;
      double p_ap_dot = 0;

      spmv_time += spmv(Ap, A, p);

      p_ap_dot = dot(Ap, p);

      if (p_ap_dot < brkdown_tol) {
        if (p_ap_dot < 0) {
          std::cerr << "miniFE::cg_solve ERROR, numerical breakdown!"
                    << std::endl;
          return num_iters;
        } else
          brkdown_tol = 0.1 * p_ap_dot;
      }
      alpha = rtrans / p_ap_dot;

      axpby(x, one, x, alpha, p);
      axpby(r, one, r, -alpha, Ap);
      num_iters = k;
    }

    // Compute SPMV Bytes and Flops
    double spmv_bytes = A.num_rows() * sizeof(int64_t) +
                        A.nnz() * sizeof(int64_t) + A.nnz() * sizeof(double) +
                        A.nnz() * sizeof(double) +
                        A.num_rows() * sizeof(double);
    double spmv_flops = A.nnz() * 2;

    double GB = (spmv_bytes) / 1024 / 1024 / 1024;

    printf("SPMV \n");
    printf("KK : SPMV : Data Transfered = %f GBs\n", GB);
    printf("KK: SPMV Performance: %lf GFlop/s %lf GB/s \n",
           1e-9 * (spmv_flops * (num_iters + 1)) / spmv_time,
           (1.0 / 1024 / 1024 / 1024) * (spmv_bytes * (num_iters + 1)) /
               spmv_time);

    return num_iters;
  }

#if defined(KOKKOS_ENABLE_OPENMPTARGET)
  template <class VType, class AType>
  int cg_solve_ompt(VType y, AType A, VType b, int max_iter, double tolerance) {
    int myproc    = 0;
    int num_iters = 0;

    double normr     = 0;
    double rtrans    = 0;
    double oldrtrans = 0;

    double spmv_time = 0.;

    int64_t print_freq = max_iter / 10;
    if (print_freq > 50) print_freq = 50;
    if (print_freq < 1) print_freq = 1;
    VType x("x", b.extent(0));
    VType r("r", x.extent(0));
    VType r_("r_", x.extent(0));
    VType p("r", x.extent(0));  // Needs to be global
    VType Ap("r", x.extent(0));
    double one  = 1.0;
    double zero = 0.0;
    axpby_ompt(p, one, x, zero, x);

    spmv_time += spmv_ompt(Ap, A, p);
    axpby_ompt(r, one, b, -one, Ap);

    rtrans = dot_ompt(r, r);

    normr = std::sqrt(rtrans);

    if (myproc == 0) {
      std::cout << "Initial Residual = " << normr << std::endl;
    }

    double brkdown_tol = std::numeric_limits<double>::epsilon();

    for (int64_t k = 1; k <= max_iter && normr > tolerance; ++k) {
      if (k == 1) {
        axpby_ompt(p, one, r, zero, r);
      } else {
        oldrtrans   = rtrans;
        rtrans      = dot_ompt(r, r);
        double beta = rtrans / oldrtrans;
        axpby_ompt(p, one, r, beta, p);
      }

      normr = std::sqrt(rtrans);

      double alpha    = 0;
      double p_ap_dot = 0;

      spmv_time += spmv_ompt(Ap, A, p);

      p_ap_dot = dot_ompt(Ap, p);

      if (p_ap_dot < brkdown_tol) {
        if (p_ap_dot < 0) {
          std::cerr << "miniFE::cg_solve ERROR, numerical breakdown!"
                    << std::endl;
          return num_iters;
        } else
          brkdown_tol = 0.1 * p_ap_dot;
      }
      alpha = rtrans / p_ap_dot;

      axpby_ompt(x, one, x, alpha, p);
      axpby_ompt(r, one, r, -alpha, Ap);
      num_iters = k;
    }

    // Compute SPMV Bytes and Flops
    double spmv_bytes = A.num_rows() * sizeof(int64_t) +
                        A.nnz() * sizeof(int64_t) + A.nnz() * sizeof(double) +
                        A.nnz() * sizeof(double) +
                        A.num_rows() * sizeof(double);
    double spmv_flops = A.nnz() * 2;

    double GB = (spmv_bytes) / 1024 / 1024 / 1024;

    printf("SPMV \n");
    printf("OMPT : SPMV : Data Transfered = %f GBs\n", GB);
    printf("OMPT: SPMV Performance: %lf GFlop/s %lf GB/s \n",
           1e-9 * (spmv_flops * (num_iters + 1)) / spmv_time,
           (1.0 / 1024 / 1024 / 1024) * (spmv_bytes * (num_iters + 1)) /
               spmv_time);
    return num_iters;
  }
#endif

  void run_kk_test() {
    Kokkos::Timer timer;
    int num_iters = cg_solve_kk(y, A, x, max_iter, tolerance);
    double time   = timer.seconds();

    // Compute Bytes and Flops
    double spmv_bytes = A.num_rows() * sizeof(int64_t) +
                        A.nnz() * sizeof(int64_t) + A.nnz() * sizeof(double) +
                        A.nnz() * sizeof(double) +
                        A.num_rows() * sizeof(double);

    double dot_bytes   = x.extent(0) * sizeof(double) * 2;
    double axpby_bytes = x.extent(0) * sizeof(double) * 3;

    double GB = (spmv_bytes + dot_bytes + axpby_bytes) / 1024 / 1024 / 1024;
    printf("Data Transferred = %f GBs\n", GB);

    double spmv_flops  = A.nnz() * 2;
    double dot_flops   = x.extent(0) * 2;
    double axpby_flops = x.extent(0) * 3;

    int spmv_calls  = 1 + num_iters;
    int dot_calls   = num_iters;
    int axpby_calls = 2 + num_iters * 3;

    // KK info
    printf("CGSolve \n");
    printf("KK: CGSolve for 3D (%i %i %i); %i iterations; %lf time\n", N, N, N,
           num_iters, time);
    printf(
        "KK: Performance: %lf GFlop/s %lf GB/s (Calls SPMV: %i Dot: %i "
        "AXPBY: %i\n",
        1e-9 *
            (spmv_flops * spmv_calls + dot_flops * dot_calls +
             axpby_flops * axpby_calls) /
            time,
        (1.0 / 1024 / 1024 / 1024) *
            (spmv_bytes * spmv_calls + dot_bytes * dot_calls +
             axpby_bytes * axpby_calls) /
            time,
        spmv_calls, dot_calls, axpby_calls);
  }

#if defined(KOKKOS_ENABLE_OPENMPTARGET)
  void run_ompt_test() {
    Kokkos::Timer timer;
    int num_iters = cg_solve_ompt(y, A, x, max_iter, tolerance);
    double time   = timer.seconds();

    // Compute Bytes and Flops
    double spmv_bytes = A.num_rows() * sizeof(int64_t) +
                        A.nnz() * sizeof(int64_t) + A.nnz() * sizeof(double) +
                        A.nnz() * sizeof(double) +
                        A.num_rows() * sizeof(double);

    double dot_bytes   = x.extent(0) * sizeof(double) * 2;
    double axpby_bytes = x.extent(0) * sizeof(double) * 3;

    double spmv_flops  = A.nnz() * 2;
    double dot_flops   = x.extent(0) * 2;
    double axpby_flops = x.extent(0) * 3;

    int spmv_calls  = 1 + num_iters;
    int dot_calls   = num_iters;
    int axpby_calls = 2 + num_iters * 3;

    // OMPT info
    printf("CGSolve \n");
    printf("OMPT: CGSolve for 3D (%i %i %i); %i iterations; %lf time\n", N, N,
           N, num_iters, time);
    printf(
        "OMPT: Performance: %lf GFlop/s %lf GB/s (Calls SPMV: %i Dot: %i "
        "AXPBY: %i\n",
        1e-9 *
            (spmv_flops * spmv_calls + dot_flops * dot_calls +
             axpby_flops * axpby_calls) /
            time,
        (1.0 / 1024 / 1024 / 1024) *
            (spmv_bytes * spmv_calls + dot_bytes * dot_calls +
             axpby_bytes * axpby_calls) /
            time,
        spmv_calls, dot_calls, axpby_calls);
  }
#endif

  void run_test() {
    printf("*******Kokkos***************\n");
    printf("Kokkos::ExecutionSpace = %s\n",
           typeid(Kokkos::DefaultExecutionSpace).name());
    run_kk_test();
#if defined(KOKKOS_ENABLE_OPENMPTARGET)
    printf("*******OpenMPTarget***************\n");
    run_ompt_test();
#endif
    printf("\n");
  }
};

