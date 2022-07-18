#include "optimizer.h"

namespace graphiler {
void backpropemitter(std::shared_ptr<MPDFGAnnotation> &mpdfg) {
    //std::shared_ptr<torch::jit::Graph> &graph) {
    std::shared_ptr<torch::jit::Graph> graph_copy=mpdfg->DFG->copy();
    auto grad_spec = differentiate(graph_copy);
    printf("\ngrad_spec.f: \n");
    printf("%s\n",(grad_spec.f)->toString());
    printf("\ngrad_spec.df: \n");
    printf("%s\n",(grad_spec.df)->toString());
}

}