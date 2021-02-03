/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <Kokkos_Core.hpp>

// Estimate memory pool size
size_t estimate_required_memory(int n) {
  assert(n >= 0);
  auto nl = static_cast<size_t>(n);
  return (nl + 1) * (nl + 1) * 2000;
}

template <class Scheduler>
struct FibonacciTask {
  using value_type  = int;
  using future_type = Kokkos::BasicFuture<int, Scheduler>;

  int n;
  future_type fn_1;
  future_type fn_2;

  KOKKOS_INLINE_FUNCTION
  explicit FibonacciTask(int num) noexcept : n(num) {}

  template <class TeamMember>
  KOKKOS_INLINE_FUNCTION void operator()(TeamMember &member, int &result) {
    auto &scheduler = member.scheduler();
    if (n < 2) {
      // Final task reached max traversal depth
      result = n;
    } else if (!fn_1.is_null() && !fn_2.is_null()) {
      // Results from child tasks are ready
      result = fn_1.get() + fn_2.get();
    } else {
      // Spawn child tasks
      fn_1 = Kokkos::task_spawn(Kokkos::TaskSingle(scheduler),
                                FibonacciTask{n - 1});
      fn_2 = Kokkos::task_spawn(Kokkos::TaskSingle(scheduler),
                                FibonacciTask{n - 2});

      // Create an aggregate predecessor for respawn
      Kokkos::BasicFuture<void, Scheduler> fib_array[] = {fn_1, fn_2};
      auto f_all = scheduler.when_all(fib_array, 2);

      // Respawn this task with `f_all` as predecessor
      Kokkos::respawn(this, f_all);
    }
  }
};

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  int n = 10;  // Fib number to compute

  using scheduler_type = Kokkos::TaskScheduler<Kokkos::DefaultExecutionSpace>;
  using memory_space   = typename scheduler_type::memory_space;
  using memory_pool    = typename scheduler_type::memory_pool;

  auto mpool     = memory_pool(memory_space(), estimate_required_memory(n));
  auto scheduler = scheduler_type(mpool);

  Kokkos::BasicFuture<int, scheduler_type> result;

  // launch the root task from the host
  result = Kokkos::host_spawn(Kokkos::TaskSingle(scheduler),
                              FibonacciTask<scheduler_type>{n});

  // Wait on all tasks submitted to the scheduler to be done
  Kokkos::wait(scheduler);
  printf("  Success! Fibonacci(%d) = %d\n", n, result.get());
  return 0;
}
