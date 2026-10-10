// Arm C (STABLE_TORCH_LIBRARY): same bodies, stable-ABI types.
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
using torch::stable::Tensor;

Tensor my_op_stable(Tensor x) { return x; }

// const_data_ptr() is untyped here, so the checks are the only guard.
Tensor my_op_work_stable(Tensor x) {
  STD_TORCH_CHECK(x.scalar_type() == torch::headeronly::ScalarType::Float,
                  "my_op_work: expected float32");
  STD_TORCH_CHECK(x.is_contiguous(), "my_op_work: expected a contiguous tensor");
  Tensor out = torch::stable::empty_like(x);
  const float* in = static_cast<const float*>(x.const_data_ptr());
  float* o = static_cast<float*>(out.mutable_data_ptr());
  for (int64_t i = 0; i < x.numel(); ++i) o[i] = in[i] * 2.0f + 1.0f;
  return out;
}

void tensor_in_1_stable(Tensor) {}
void tensor_in_2_stable(Tensor, Tensor) {}
void tensor_in_3_stable(Tensor, Tensor, Tensor) {}
void tensor_in_4_stable(Tensor, Tensor, Tensor, Tensor) {}
void tensor_in_5_stable(Tensor, Tensor, Tensor, Tensor, Tensor) {}

void int_in_1_stable(int64_t) {}
void int_in_2_stable(int64_t, int64_t) {}
void int_in_3_stable(int64_t, int64_t, int64_t) {}
void int_in_4_stable(int64_t, int64_t, int64_t, int64_t) {}
void int_in_5_stable(int64_t, int64_t, int64_t, int64_t, int64_t) {}

Tensor tensor_out_1_stable(Tensor x) { return x; }
std::tuple<Tensor, Tensor> tensor_out_2_stable(Tensor x) { return {x, x}; }
std::tuple<Tensor, Tensor, Tensor> tensor_out_3_stable(Tensor x) { return {x, x, x}; }
std::tuple<Tensor, Tensor, Tensor, Tensor> tensor_out_4_stable(Tensor x) { return {x, x, x, x}; }
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> tensor_out_5_stable(Tensor x) { return {x, x, x, x, x}; }

int64_t int_out_1_stable(int64_t x) { return x; }
std::tuple<int64_t, int64_t> int_out_2_stable(int64_t x) { return {x, x}; }
std::tuple<int64_t, int64_t, int64_t> int_out_3_stable(int64_t x) { return {x, x, x}; }
std::tuple<int64_t, int64_t, int64_t, int64_t> int_out_4_stable(int64_t x) { return {x, x, x, x}; }
std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t> int_out_5_stable(int64_t x) { return {x, x, x, x, x}; }

STABLE_TORCH_LIBRARY(pvd_stable, m) {
  m.def("my_op(Tensor(a) x) -> Tensor(a)");
  m.def("my_op_work(Tensor x) -> Tensor");
  m.def("tensor_in_1(Tensor a1) -> ()");
  m.def("tensor_in_2(Tensor a1, Tensor a2) -> ()");
  m.def("tensor_in_3(Tensor a1, Tensor a2, Tensor a3) -> ()");
  m.def("tensor_in_4(Tensor a1, Tensor a2, Tensor a3, Tensor a4) -> ()");
  m.def("tensor_in_5(Tensor a1, Tensor a2, Tensor a3, Tensor a4, Tensor a5) -> ()");
  m.def("int_in_1(int a1) -> ()");
  m.def("int_in_2(int a1, int a2) -> ()");
  m.def("int_in_3(int a1, int a2, int a3) -> ()");
  m.def("int_in_4(int a1, int a2, int a3, int a4) -> ()");
  m.def("int_in_5(int a1, int a2, int a3, int a4, int a5) -> ()");
  m.def("tensor_out_1(Tensor(a) x) -> Tensor(a)");
  m.def("tensor_out_2(Tensor(a) x) -> (Tensor(a), Tensor(a))");
  m.def("tensor_out_3(Tensor(a) x) -> (Tensor(a), Tensor(a), Tensor(a))");
  m.def("tensor_out_4(Tensor(a) x) -> (Tensor(a), Tensor(a), Tensor(a), Tensor(a))");
  m.def("tensor_out_5(Tensor(a) x) -> (Tensor(a), Tensor(a), Tensor(a), Tensor(a), Tensor(a))");
  m.def("int_out_1(int x) -> int");
  m.def("int_out_2(int x) -> (int, int)");
  m.def("int_out_3(int x) -> (int, int, int)");
  m.def("int_out_4(int x) -> (int, int, int, int)");
  m.def("int_out_5(int x) -> (int, int, int, int, int)");
}
STABLE_TORCH_LIBRARY_IMPL(pvd_stable, CompositeExplicitAutograd, m) {
  m.impl("my_op", TORCH_BOX(&my_op_stable));
  m.impl("my_op_work", TORCH_BOX(&my_op_work_stable));
  m.impl("tensor_in_1", TORCH_BOX(&tensor_in_1_stable));
  m.impl("tensor_in_2", TORCH_BOX(&tensor_in_2_stable));
  m.impl("tensor_in_3", TORCH_BOX(&tensor_in_3_stable));
  m.impl("tensor_in_4", TORCH_BOX(&tensor_in_4_stable));
  m.impl("tensor_in_5", TORCH_BOX(&tensor_in_5_stable));
  m.impl("tensor_out_1", TORCH_BOX(&tensor_out_1_stable));
  m.impl("tensor_out_2", TORCH_BOX(&tensor_out_2_stable));
  m.impl("tensor_out_3", TORCH_BOX(&tensor_out_3_stable));
  m.impl("tensor_out_4", TORCH_BOX(&tensor_out_4_stable));
  m.impl("tensor_out_5", TORCH_BOX(&tensor_out_5_stable));
}
// The stable registration macro parses its dispatch-key token as a string;
// "Undefined" is the spelling it accepts for the CatchAll key.
STABLE_TORCH_LIBRARY_IMPL(pvd_stable, Undefined, m) {
  m.impl("int_in_1", TORCH_BOX(&int_in_1_stable));
  m.impl("int_in_2", TORCH_BOX(&int_in_2_stable));
  m.impl("int_in_3", TORCH_BOX(&int_in_3_stable));
  m.impl("int_in_4", TORCH_BOX(&int_in_4_stable));
  m.impl("int_in_5", TORCH_BOX(&int_in_5_stable));
  m.impl("int_out_1", TORCH_BOX(&int_out_1_stable));
  m.impl("int_out_2", TORCH_BOX(&int_out_2_stable));
  m.impl("int_out_3", TORCH_BOX(&int_out_3_stable));
  m.impl("int_out_4", TORCH_BOX(&int_out_4_stable));
  m.impl("int_out_5", TORCH_BOX(&int_out_5_stable));
}
