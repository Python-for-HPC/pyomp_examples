.PHONY: all

all: heat_data_reg.x heat_data_reg-transpose.x

heat_data_reg.x: heat_data_reg.c
	clang -O3 -fopenmp -fopenmp-targets=nvptx64 -fopenmp-target-new-runtime heat_data_reg.c -o heat_data_reg.x -lm

heat_data_reg-transpose.x: heat_data_reg-transpose.c
	clang -O3 -fopenmp -fopenmp-targets=nvptx64 -fopenmp-target-new-runtime heat_data_reg-transpose.c -o heat_data_reg-transpose.x -lm

clean:
	rm -f *.ll *.o *.bc *.s *.i *.cubin a.out* *.x
