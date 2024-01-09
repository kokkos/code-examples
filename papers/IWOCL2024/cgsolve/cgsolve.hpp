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

//#define USE_TPL


#ifdef USE_TPL
#include <KokkosSparse_spmv.hpp>
#include <KokkosBlas.hpp>
#endif
using INT_TYPE = int64_t;

//#define ONLY_DO_SPMV

#ifdef ONLY_DO_SPMV
double time_spmv_kk = 0.;
double time_spmv_sycl = 0.;
#endif

/*
 * There is a bug in the clang OpenMP implementation wherein if the `dot`
 * routine is called with the same View as the 1st and 2nd parameter, the norm
 * fails to converge. Hence for such cases we perform the dot product on the
 * host.
 */
struct cgsolve {
    int N, max_iter;
    double tolerance;
    Kokkos::View<double *> y, x;
    CrsMatrix<Kokkos::DefaultExecutionSpace::memory_space> A;

    static_assert(std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::Experimental::SYCL>);

    cgsolve(int N_, int max_iter_in, double tolerance_in)
        : N(N_), max_iter(max_iter_in), tolerance(tolerance_in) {
        CrsMatrix<Kokkos::HostSpace> h_A = Impl::generate_miniFE_matrix(N);
        Kokkos::View<double *, Kokkos::HostSpace> h_x =
            Impl::generate_miniFE_vector(N);

        Kokkos::View<INT_TYPE *> row_ptr("row_ptr", h_A.row_ptr.extent(0));
        Kokkos::View<INT_TYPE *> col_idx("col_idx", h_A.col_idx.extent(0));
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

#ifdef USE_TPL
    template <class YType, class AType, class XType>
    void spmv(YType y, AType A, XType x) {
      KokkosSparse::CrsMatrix<
        double, INT_TYPE,
        Kokkos::Device<Kokkos::DefaultExecutionSpace,
                       typename Kokkos::DefaultExecutionSpace::memory_space>,
        void, INT_TYPE>
        matrix("A",           // const std::string& /* label */,
               A.num_rows(),  // const OrdinalType nrows,
               A.num_rows(),  // const OrdinalType ncols,
               A.nnz(),       // const size_type annz,
               A.values,      // const values_type& vals,
               A.row_ptr,     // const row_map_type& rowmap,
               A.col_idx);    // const index_type& cols)
#ifdef ONLY_DO_SPMV
        Kokkos::fence();
        Kokkos::Timer timer_spmv;
#endif
      KokkosSparse::spmv("N", 1.0, matrix, x, 0.0, y);
#ifdef ONLY_DO_SPMV
        Kokkos::fence();
        time_spmv_kk += timer_spmv.seconds();
#endif
    }
#else
    template <class YType, class AType, class XType>
    void spmv(YType y, AType A, XType x) {
        int rows_per_team = 32; //10240 too large, 5120 OK
        int team_size = 16;
        int vector_size = 4;
        INT_TYPE nrows = y.extent(0);
	//std::cout << nrows << std::endl;
#ifdef ONLY_DO_SPMV
	Kokkos::fence();
	Kokkos::Timer timer_spmv;
#endif
	Kokkos::parallel_for(
            "SPMV",
            Kokkos::TeamPolicy<>((nrows + rows_per_team - 1) / rows_per_team,
                                 team_size, vector_size),
            KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
                const INT_TYPE first_row = team.league_rank() * rows_per_team;
                const INT_TYPE last_row = first_row + rows_per_team < nrows
                                             ? first_row + rows_per_team
                                             : nrows;
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, first_row, last_row),
                    [&](const INT_TYPE row) {
                        const INT_TYPE row_start = A.row_ptr(row);
                        const INT_TYPE row_length =
                            A.row_ptr(row + 1) - row_start;

                        double y_row;
                        Kokkos::parallel_reduce(
                            Kokkos::ThreadVectorRange(team, row_length),
                            [=](const INT_TYPE i, double &sum) {
                                sum += A.values(i + row_start) *
                                       x(A.col_idx(i + row_start));
                            },
                            y_row);
                        y(row) = y_row;
                    });
            });
#ifdef ONLY_DO_SPMV
	Kokkos::fence();
	time_spmv_kk += timer_spmv.seconds();
#endif
    }
#endif

#if defined(KOKKOS_ENABLE_SYCL)
    template <class YType, class AType, class XType>
    void spmv_sycl(sycl::queue q, YType y, AType A, XType x) {
        int rows_per_team = 32;
        int team_size = 16;
        INT_TYPE nrows = y.extent(0);

        auto row_ptr = A.row_ptr.data();
        auto values = A.values.data();
        auto col_idx = A.col_idx.data();
        auto xp = x.data();
        auto yp = y.data();

        INT_TYPE n = (nrows + rows_per_team - 1) / rows_per_team;
#ifdef ONLY_DO_SPMV
        q.wait();
        Kokkos::Timer timer_spmv;
#endif
	q.submit([&] (sycl::handler& cgh) {
         cgh.parallel_for_work_group(sycl::range<1>(n), sycl::range<1>(team_size), [=](sycl::group<1> g) {
	    const INT_TYPE first_row = g.get_group_id(0) * rows_per_team;
            const INT_TYPE last_row = first_row + rows_per_team < nrows
                                             ? first_row + rows_per_team
                                             : nrows;
	    g.parallel_for_work_item(sycl::range<1>(last_row-first_row), [&](sycl::h_item<1> item) {
              const INT_TYPE row = item.get_local_id(0) + first_row;
	      const INT_TYPE row_start = row_ptr[row];
              const INT_TYPE row_length = row_ptr[row + 1] - row_start;
              double y_row = 0.;
              for (INT_TYPE i = 0; i < row_length; ++i)
                y_row += values[i + row_start] * xp[col_idx[i + row_start]];
              yp[row] = y_row;
            });
          });
        });
#ifdef ONLY_DO_SPMV
        q.wait();	
	time_spmv_sycl += timer_spmv.seconds();
#endif
    }
#endif

    template <class YType, class XType>
    double dot(YType y, XType x) {
        double result;
        Kokkos::parallel_reduce(
            "DOT", y.extent(0),
            KOKKOS_LAMBDA(const INT_TYPE &i, double &lsum) {
                lsum += y(i) * x(i);
            },
            result);
        return result;
    }

#if defined(KOKKOS_ENABLE_SYCL)
    template <class YType, class XType>
    double dot_sycl(sycl::queue q, YType y, XType x) {
        int n = y.extent(0);
        auto xp = x.data();
        auto yp = y.data();

	double result = 0.;
	sycl::buffer<double> result_buffer{&result, 1};
	q.submit([&](sycl::handler& cgh) {
	  auto reducer = sycl::reduction(result_buffer, cgh, sycl::plus<>());
	  cgh.parallel_for(sycl::range<1>(n), reducer, 
            [=](sycl::id<1> idx, auto&sum) {
              int i = idx;
              sum += yp[i]*xp[i];
            });
	});
	return result_buffer.get_host_access()[0];  
    }
#endif

#if defined(KOKKOS_ENABLE_SYCL) && !defined(KOKKOS_COMPILER_NVHPC)
    template <class VType>
    double dot_sycl_bug(VType r) {
        double result = 0.;
        auto h_r = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r);
        for (int i = 0; i < h_r.extent(0); ++i) result += h_r(i) * h_r(i);

        return result;
    }
#endif

    template <class ZType, class YType, class XType>
    void axpby(ZType z, double alpha, XType x, double beta, YType y) {
        INT_TYPE n = z.extent(0);
        Kokkos::parallel_for(
            "AXPBY", n,
            KOKKOS_LAMBDA(const int &i) { z(i) = alpha * x(i) + beta * y(i); });
	//Kokkos::fence();
    }

#if defined(KOKKOS_ENABLE_SYCL)
    template <class ZType, class YType, class XType>
    void axpby_sycl(sycl::queue& q, ZType z, double alpha, XType x, double beta, YType y) {
        INT_TYPE n = z.extent(0);
        auto xp = x.data();
        auto yp = y.data();
        auto zp = z.data();

	q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
          int i = idx;
          zp[i] = alpha * xp[i] + beta * yp[i];
        });
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
        int myproc = 0;
        int num_iters = 0;

        double normr = 0;
        double rtrans = 0;
        double oldrtrans = 0;

        INT_TYPE print_freq = max_iter / 10;
        if (print_freq > 50) print_freq = 50;
        if (print_freq < 1) print_freq = 1;
        VType x("x", b.extent(0));
        VType r("r", x.extent(0));
        VType p("r", x.extent(0));  // Needs to be global
        VType Ap("r", x.extent(0));
        double one = 1.0;
        double zero = 0.0;
        axpby(p, one, x, zero, x);

        spmv(Ap, A, p);
        axpby(r, one, b, -one, Ap);

#if defined(KOKKOS_ENABLE_SYCL) && defined(KOKKOS_ARCH_VEGA90A)
        rtrans = dot_sycl_bug(r);
#else
        rtrans = dot(r, r);
#endif

        normr = std::sqrt(rtrans);

        if (myproc == 0) {
            //std::cout << "Initial Residual = " << normr << std::endl;
        }

        double brkdown_tol = std::numeric_limits<double>::epsilon();

        for (INT_TYPE k = 1; k <= max_iter && normr > tolerance; ++k) {
            if (k == 1) {
                axpby(p, one, r, zero, r);
            } else {
                oldrtrans = rtrans;
#if defined(KOKKOS_ENABLE_SYCL) && defined(KOKKOS_ARCH_VEGA90A)
                rtrans = dot_sycl_bug(r);
#else
                rtrans = dot(r, r);
#endif
                double beta = rtrans / oldrtrans;
                axpby(p, one, r, beta, p);
            }

            normr = std::sqrt(rtrans);

            double alpha = 0;
            double p_ap_dot = 0;

            spmv(Ap, A, p);

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
	Kokkos::fence();
        return num_iters;
    }

#if defined(KOKKOS_ENABLE_SYCL)
    template <class VType, class AType>
    int cg_solve_sycl(VType solution, AType A, VType b, int max_iter,
                      double tolerance) {
        int myproc = 0;
        int num_iters = 0;

	sycl::queue q{sycl::property::queue::in_order()};

	double normr = 0;
        double rtrans = 0;
        double oldrtrans = 0;

        INT_TYPE print_freq = max_iter / 10;
        if (print_freq > 50) print_freq = 50;
        if (print_freq < 1) print_freq = 1;
        VType x("x", b.extent(0));
        VType r("r", x.extent(0));
        VType p("r", x.extent(0));  // Needs to be global
        VType Ap("r", x.extent(0));
        double one = 1.0;
        double zero = 0.0;
        axpby_sycl(q, p, one, x, zero, x);

        spmv_sycl(q, Ap, A, p);
        axpby_sycl(q, r, one, b, -one, Ap);

#if defined(KOKKOS_COMPILER_CLANG)
        rtrans = dot_sycl_bug(r);
#else
        rtrans = dot_sycl(q, r, r);
#endif

        normr = std::sqrt(rtrans);

        if (myproc == 0) {
            //std::cout << "Initial Residual = " << normr << std::endl;
        }

        double brkdown_tol = std::numeric_limits<double>::epsilon();

        for (INT_TYPE k = 1; k <= max_iter && normr > tolerance; ++k) {
            if (k == 1) {
                axpby_sycl(q, p, one, r, zero, r);
            } else {
                oldrtrans = rtrans;
#if defined(KOKKOS_COMPILER_CLANG)
                rtrans = dot_sycl_bug(r);
#else
                rtrans = dot_sycl(q, r, r);
#endif
                double beta = rtrans / oldrtrans;
                axpby_sycl(q, p, one, r, beta, p);
            }

            normr = std::sqrt(rtrans);

            double alpha = 0;
            double p_ap_dot = 0;

            spmv_sycl(q, Ap, A, p);

            p_ap_dot = dot_sycl(q, Ap, p);

            if (p_ap_dot < brkdown_tol) {
                if (p_ap_dot < 0) {
                    std::cerr << "miniFE::cg_solve ERROR, numerical breakdown!"
                              << std::endl;
                    return num_iters;
                } else
                    brkdown_tol = 0.1 * p_ap_dot;
            }
            alpha = rtrans / p_ap_dot;

            axpby_sycl(q, x, one, x, alpha, p);
            axpby_sycl(q, r, one, r, -alpha, Ap);
            num_iters = k;
        }
	q.wait();
        return num_iters;
    }
#endif

    void run_kk_test() {
#ifdef USE_TPL
	std::cout << "Using oneMKL\n" << std::endl;
#endif
#ifdef ONLY_DO_SPMV
	time_spmv_kk = 0;
#endif

        Kokkos::Timer timer;
        int num_iters = cg_solve_kk(y, A, x, max_iter, tolerance);
        double time = timer.seconds();

        // Compute Bytes and Flops
        double spmv_bytes =
            A.num_rows() * sizeof(INT_TYPE) + A.nnz() * sizeof(INT_TYPE) +
            A.nnz() * sizeof(double) + A.nnz() * sizeof(double) +
            A.num_rows() * sizeof(double);

        double dot_bytes = x.extent(0) * sizeof(double) * 2;
        double axpby_bytes = x.extent(0) * sizeof(double) * 3;

        double GB = (spmv_bytes + dot_bytes + axpby_bytes) / 1024 / 1024 / 1024;
        //printf("Data Transferred = %f GBs\n", GB);

        double spmv_flops = A.nnz() * 2;
        double dot_flops = x.extent(0) * 2;
        double axpby_flops = x.extent(0) * 3;

        int spmv_calls = 1 + num_iters;
        int dot_calls = 2*num_iters;
        int axpby_calls = 2 + num_iters * 3;

#ifdef ONLY_DO_SPMV
        std::cout << A.num_rows() << " KK(SPMV) " << ((1.0 / 1024 / 1024 / 1024) * spmv_bytes * spmv_calls)/time_spmv_kk << "GB/s ";
#else
        // KK info
        //printf("KK: CGSolve for 3D (%i %i %i); %i iterations; %lf time\n", N, N,
        //       N, num_iters, time);
        printf(
            "%ld KK: Performance: %lf GFlop/s %lf GB/s (Calls SPMV: %i Dot: %i "
            "AXPBY: %i\n",A.num_rows(),
            1e-9 *
                (spmv_flops * spmv_calls + dot_flops * dot_calls +
                 axpby_flops * axpby_calls) /
                time,
            (1.0 / 1024 / 1024 / 1024) *
                (spmv_bytes * spmv_calls + dot_bytes * dot_calls +
                 axpby_bytes * axpby_calls) /
                time,
            spmv_calls, dot_calls, axpby_calls);
#endif
    }

#if defined(KOKKOS_ENABLE_SYCL)
    void run_sycl_test() {
        Kokkos::Timer timer;
        int num_iters = cg_solve_sycl(y, A, x, max_iter, tolerance);
        double time = timer.seconds();

        // Compute Bytes and Flops
        double spmv_bytes =
            A.num_rows() * sizeof(INT_TYPE) + A.nnz() * sizeof(INT_TYPE) +
            A.nnz() * sizeof(double) + A.nnz() * sizeof(double) +
            A.num_rows() * sizeof(double);

        double dot_bytes = x.extent(0) * sizeof(double) * 2;
        double axpby_bytes = x.extent(0) * sizeof(double) * 3;

        double spmv_flops = A.nnz() * 2;
        double dot_flops = x.extent(0) * 2;
        double axpby_flops = x.extent(0) * 3;

        int spmv_calls = 1 + num_iters;
        int dot_calls = 2*num_iters;
        int axpby_calls = 2 + num_iters * 3;

#ifdef ONLY_DO_SPMV
        std::cout << "SYCL(SPMV) " << ((1.0 / 1024 / 1024 / 1024) * spmv_bytes * spmv_calls)/time_spmv_sycl << "GB/s\n";
#else
	// SYCL info
        //printf("SYCL: CGSolve for 3D (%i %i %i); %i iterations; %lf time\n", N,
        //       N, N, num_iters, time);
        printf(
            "SYCL: Performance: %lf GFlop/s %lf GB/s (Calls SPMV: %i Dot: %i "
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
#endif
    }
#endif

    void run_test() {
        //printf("*******Kokkos***************\n");
        //printf("Kokkos::ExecutionSpace = %s\n",
        //       typeid(Kokkos::DefaultExecutionSpace).name());
        run_kk_test();
#if defined(KOKKOS_ENABLE_SYCL)
        //printf("*******OpenMPTarget***************\n");
        run_sycl_test();
#endif
    }
};

