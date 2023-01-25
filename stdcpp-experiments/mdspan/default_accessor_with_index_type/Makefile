KOKKOS_DEVICES=Cuda
KOKKOS_CUDA_OPTIONS=enable_lambda
KOKKOS_OPTIONS=impl_disable_bundled_mdspan
KOKKOS_ARCH = "Volta70"
KOKKOS_PATH = ${HOME}/Kokkos/kokkos
MDSPAN_PATH = ${HOME}/Kokkos/mdspan

MAKEFILE_PATH := $(subst Makefile,,$(abspath $(lastword $(MAKEFILE_LIST))))

SRC = $(wildcard $(MAKEFILE_PATH)*.cpp)
HEADERS = $(wildcard $(MAKEFILE_PATH)*.hpp)

vpath %.cpp $(sort $(dir $(SRC)))

default: build
	echo "Start Build"

ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
EXE = test.cuda
else
CXX = g++
EXE = test.host
endif

CXXFLAGS ?= -O3 -g
override CXXFLAGS += -I$(MAKEFILE_PATH) -I$(MDSPAN_PATH)/include

ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
  override CXXFLAGS += -expt-relaxed-constexpr
endif

DEPFLAGS = -M
LINK = ${CXX}
LINKFLAGS =

OBJ = $(notdir $(SRC:.cpp=.o))
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean
	rm -f *.o *.cuda *.host

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS) $(HEADERS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $< -o $(notdir $@)
