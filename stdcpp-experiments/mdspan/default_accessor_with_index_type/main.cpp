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

#include<Kokkos_Core.hpp>
#include<experimental/mdspan>
#include<type_traits>


template <class ElementType, class IndexT>
struct indext_accessor {
  using offset_policy = indext_accessor;
  using element_type = ElementType;
  using reference = ElementType&;
  using data_handle_type = ElementType*;

  MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr indext_accessor() noexcept = default;

  MDSPAN_TEMPLATE_REQUIRES(
    class OtherElementType, class OtherIndexT, 
    /* requires */ (
      _MDSPAN_TRAIT(std::is_convertible, OtherElementType(*)[], element_type(*)[])
    )
  )
  MDSPAN_INLINE_FUNCTION
  constexpr indext_accessor(indext_accessor<OtherElementType, OtherIndexT>) noexcept {}

  MDSPAN_INLINE_FUNCTION
  constexpr data_handle_type
  offset(data_handle_type p, IndexT i) const noexcept {
    return p + i;
  }

  MDSPAN_FORCE_INLINE_FUNCTION
  constexpr reference access(data_handle_type p, IndexT i) const noexcept {
    return p[i];
  }
};

template<class INDEXT, class OFFSETT>
void run_test(INDEXT R, Kokkos::View<int**> a_v, bool do_print) {
     using ext_t = std::experimental::extents<INDEXT, std::experimental::dynamic_extent>;
     using map_t = std::experimental::layout_left::mapping<ext_t>;
     using acc_t = indext_accessor<int, OFFSETT>;
     using mdspan_t = std::experimental::mdspan<int, ext_t, std::experimental::layout_left, acc_t>;

     INDEXT N = a_v.extent(0);
     int* a = a_v.data();
     Kokkos::Timer timer;

     Kokkos::parallel_for(N, KOKKOS_LAMBDA(INDEXT i0) {
       int values[8];
       // CUDA before 12.0 with GCC 11 as host chokes on the CTAD here
       #ifdef KOKKOS_ENABLE_CUDA
       mdspan_t a(values,map_t(ext_t(8)),acc_t());
       #else
       std::experimental::mdspan a(values, map_t(ext_t(8)), acc_t());
       #endif
       for(INDEXT i=0; i<8; i++)
         a(i) = 0;

       // On NVIDIA all variants are the same speed if I use double here
       // But clang will optimize away the loop if I use int for val
       // i.e. the timing becomes independent of R
       #ifdef KOKKOS_ENABLE_CUDA
       int
       #else
       double
       #endif
          val = 2;
       for(INDEXT r=0; r<R; r++){
         for(INDEXT i=0; i<8; i++) {
           // do something to not optimize away loop
           val += i-r;
           a(i) += val;
         }
       }
       for(INDEXT i=0; i<8; i++) {
         a_v(i0,i) = a(i);
       }
     });

     Kokkos::fence();
     double time = timer.seconds();
     if(do_print)
       printf("%i %i %lf\n",int(sizeof(INDEXT)*8), int(sizeof(OFFSETT)*8), time);
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv);
  {
     int N = (argc>1) ? atoi(argv[1]) : 1000000;
     int R = (argc>2) ? atoi(argv[2]) : 100000;

     Kokkos::View<int**> a_v("A",N,8);

     printf("IndexType OffsetType Time\n");
     run_test<size_t,size_t>(R,a_v,false);
     run_test<size_t,size_t>(R,a_v,true);
     run_test<int,size_t>(R,a_v,false);
     run_test<int,size_t>(R,a_v,true);
     run_test<int,int>(R,a_v,false);
     run_test<int,int>(R,a_v,true);
  }
  Kokkos::finalize();
}
