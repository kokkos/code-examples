include ../Makefile.inc

HEADER = dot.hpp axpby.hpp

default: build
	echo "Start Build"

EXE = test.${backend}

CXXFLAGS = -O3 -g

CXXFLAGS += ${OFFLOAD_FLAGS}
#CXXFLAGS += -foffload-lto --save-temps

LINK = ${CXX} ${CXXFLAGS}

DEPFLAGS = -M

OBJ = $(SRC:.cpp=.o)
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean
	rm -f *.o *.ompt *.cuda *.host *.openmp *.hip

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS) $(HEADER)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<
