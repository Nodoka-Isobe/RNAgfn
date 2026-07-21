// rnaeval_wrapper.c
#include "eval/eval_structure.h"
#include "eval/fold_vars.h"
#include "eval/part_func.h"
#include <string.h>

float eval_energy(const char* seq, const char* structure) {
    cut_point = -1;
    update_fold_params();
    return energy_of_struct(seq, structure);
}

float pf_eval_energy(const char* seq, const char* structure) {
    cut_point = -1;
    update_pf_params(strlen(seq));
    return pf_fold((char *)seq, (char *)structure);
}
