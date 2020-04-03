TARGETS = knn

CROSS_TOOL =
CC_CPP = $(CROSS_TOOL)g++
CC_C = $(CROSS_TOOL)gcc

CFLAGS = -std=c++14 -Werror -O2 -g -lpthread -lm

all: clean $(TARGETS)

$(TARGETS):
	$(CC_CPP) $(CFLAGS)  $@.cpp -o $@
	
clean:
	rm -f $(TARGETS)
