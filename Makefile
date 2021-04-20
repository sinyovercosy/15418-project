.PHONY: ref omp mpi

default: ref omp mpi

ref:
	$(MAKE) -C ref

omp:
	$(MAKE) -C omp

mpi:
	$(MAKE) -C mpi

clean:
	$(MAKE) -C ref clean
	$(MAKE) -C omp clean
	$(MAKE) -C mpi clean
