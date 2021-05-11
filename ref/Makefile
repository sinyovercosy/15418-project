TARGET = floyd-ref johnson-ref
OBJS += floyd-ref.o johnson-ref.o

CXX = g++
CXXFLAGS += -std=gnu++98 -Wall -Werror -Wextra -fopenmp -g -O3 -DNDEBUG
LDFLAGS += $(LIBS)

default:	$(TARGET)
all:		$(TARGET)

floyd-ref: floyd-ref.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

johnson-ref: johnson-ref.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

DEPS = $(OBJS:%.o=%.d)
-include $(DEPS)

clean:
	rm $(TARGET) $(OBJS) || true
