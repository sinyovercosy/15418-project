TARGET = floyd-mpi johnson-mpi
OBJS += floyd-mpi.o johnson-mpi.o

CXX = mpic++
CXXFLAGS += -std=gnu++11 -Wall -Werror -Wextra -pthread -fopenmp -g -O3 -DNDEBUG -MMD -MP
LDFLAGS += $(LIBS)

default:	$(TARGET)
all:		$(TARGET)

floyd-mpi: floyd-mpi.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

johnson-mpi: johnson-mpi.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

DEPS = $(OBJS:%.o=%.d)
-include $(DEPS)

clean:
	rm $(TARGET) $(OBJS) $(DEPS) || true
