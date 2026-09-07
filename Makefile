CC      = gcc
CFLAGS  = -O3 -march=native -fopenmp -Wall -Wextra
LDFLAGS = -lm

# link OpenBLAS if present (ceiling kernel)
HAS_BLAS := $(shell ldconfig -p 2>/dev/null | grep -c libopenblas)
ifneq ($(HAS_BLAS),0)
CFLAGS  += -DUSE_BLAS
LDFLAGS += -lopenblas
endif

all: matmul membench

matmul: src/01-cpu-matmul/main.c src/01-cpu-matmul/kernels.c src/01-cpu-matmul/kernels.h
	$(CC) $(CFLAGS) -o $@ src/01-cpu-matmul/main.c src/01-cpu-matmul/kernels.c $(LDFLAGS)

membench: src/01-cpu-matmul/membench.c
	$(CC) $(CFLAGS) -o $@ src/01-cpu-matmul/membench.c $(LDFLAGS)

clean:
	rm -f matmul membench

.PHONY: all clean
