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
// Kokkos is licensed under 3-clause BSD terms of use:
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

#include <Kokkos_Core.hpp>
#include <cmath>


template<class Scalar>
void benchmark(int N, int T, int P, int R, int F, int S) {
  Kokkos::View<Scalar**[3]> noise("noise",N,T);
  Kokkos::View<Scalar**> in("in",N,T*P);
  Kokkos::View<Scalar**> out("out",N,T*P);

  Kokkos::parallel_for("Test",Kokkos::TeamPolicy<>(N,T).set_scratch_size(0, Kokkos::PerTeam(S)),
    KOKKOS_LAMBDA(const typename Kokkos::TeamPolicy<>::member_type& team) {
      int l = team.league_rank();
      int t = team.team_rank();
      int ts = team.team_size();
      double min = 1.e30;
      double max = 0;
      double sum = 0;
      for(int r=0; r<R; r++) {
        auto start = Kokkos::Impl::clock_tic();
        for(int p=0; p<P; p++) {
          auto a = in(l,t+p*ts);
          auto b = out(l,t+p*ts);
          for(int f=0; f<F; f++) {
            b+=a*(f+1);
          }
          out(l,t+p*ts) = b;
        }
        auto end = Kokkos::Impl::clock_tic();
        double time = 1.*end - 1.*start;
        if(time<min) min = time;
        if(time>max) max = time;
        sum += time;
      }
      noise(l,t,0) = min;
      noise(l,t,1) = max;
      noise(l,t,2) = sum/R;
    });
  Kokkos::fence();
  auto h_noise = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), noise);

  double min = 1.e30;
  double max = 0;
  double ave = 0;
  for(int i=0; i<N; i++)
    for(int t=0; t<T; t++) {
      if(min>h_noise(i,t,0)) min = h_noise(i,t,0);
      if(max<h_noise(i,t,1)) max = h_noise(i,t,1);
      ave += h_noise(i,t,2);
    }
  ave/=1.*N*T;
  printf("Min: %e Max: %e Ave: %e\n",min,max,ave);

  int bins = 40;
  Kokkos::View<int*[3],Kokkos::HostSpace> histo("histo",bins+1);
  double factor = double(bins)/(max-min);
  for(int i=0; i<N; i++)
    for(int t=0; t<T; t++) {
      histo(int((h_noise(i,t,0)-min)*factor),0)++;
      histo(int((h_noise(i,t,1)-min)*factor),1)++;
      histo(int((h_noise(i,t,2)-min)*factor),2)++;
    }
  printf("\n");
  printf("Distributions of min, max and average measured time\n");
  printf("Time is given in multiples of global minimum time observed\n");
  printf("bin time min max ave\n");
  for(int i=0; i<bins+1; i++)
    printf("%2i %3.2lf %9i %9i %9i\n",i,(min+(1./factor)*i)/min,histo(i,0),histo(i,1),histo(i,2));
}
int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    printf("Input: N T P R F S\n");
    printf("   N (default: 10000): number of teams (CUDA blocks/HIP groups)\n");
    printf("   T (default: 256):   number of threads per team\n");
    printf("   P (default: 1):     number of work iterations per thread (2 loads, 1 store, 2F flops per iteration)\n");
    printf("   R (default: 1):     how often to repeat the work loop in each thread - each of these repeats is timed individually\n");
    printf("   F (default: 1):     flop scaling factor\n");
    printf("   S (default: 0):     scratch memory per team to restrict occupancy\n");
    printf("\n");
    printf("Narrow distribution (no oversubscription): 20000 32 1 16 1 40000\n");
    printf("Broad distribution (full overscubscription): 20000 1024 1 16 1 0\n");
    printf("\n");

    int N = argc > 1 ? atoi(argv[1]) : 10000;
    int T = argc > 2 ? atoi(argv[2]) : 256;
    int P = argc > 3 ? atoi(argv[3]) : 1;
    int R = argc > 4 ? atoi(argv[4]) : 1;
    int F = argc > 5 ? atoi(argv[5]) : 1;
    int S = argc > 6 ? atoi(argv[6]) : 0;

    benchmark<double>(N,T,P,R,F,S);
  }
  Kokkos::finalize();
}
