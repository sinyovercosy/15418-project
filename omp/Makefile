TARGET = floyd-omp johnson-omp
OBJS += floyd-omp.o johnson-omp.o

CXX = g++
CXXFLAGS += -std=gnu++98 -Wall -Werror -Wextra -pthread -fopenmp -g -O3 -DNDEBUG
LDFLAGS += $(LIBS)

default:	$(TARGET)
all:		$(TARGET)

floyd-omp: floyd-omp.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

johnson-omp: johnson-omp.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

DEPS = $(OBJS:%.o=%.d)
-include $(DEPS)

clean:
	rm $(TARGET) $(OBJS) $(DEPS) || true
