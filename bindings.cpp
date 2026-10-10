// Arms A (pybind11) and B (TORCH_LIBRARY) over the same C++ symbols.
#include <torch/extension.h>
#include <torch/library.h>

// >>> your op goes here; keep both arms calling the same function <<<

// Zero-cost body: timing it gives call overhead alone.
at::Tensor my_op(const at::Tensor& x) { return x; }

// Real work, so cost scales with numel. float32 and contiguous only.
at::Tensor my_op_work(const at::Tensor& x) {
  TORCH_CHECK(x.scalar_type() == at::kFloat, "my_op_work: expected float32");
  TORCH_CHECK(x.is_contiguous(), "my_op_work: expected a contiguous tensor");
  auto out = at::empty_like(x);
  const float* in = x.const_data_ptr<float>();
  float* o = out.mutable_data_ptr<float>();
  for (int64_t i = 0; i < x.numel(); ++i) o[i] = in[i] * 2.0f + 1.0f;
  return out;
}

// Input-side benchmark: consume 1--5 arguments and return nothing.
void tensor_in_1(const at::Tensor&) {}
void tensor_in_2(const at::Tensor&, const at::Tensor&) {}
void tensor_in_3(const at::Tensor&, const at::Tensor&, const at::Tensor&) {}
void tensor_in_4(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&) {}
void tensor_in_5(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&) {}

void int_in_1(int64_t) {}
void int_in_2(int64_t, int64_t) {}
void int_in_3(int64_t, int64_t, int64_t) {}
void int_in_4(int64_t, int64_t, int64_t, int64_t) {}
void int_in_5(int64_t, int64_t, int64_t, int64_t, int64_t) {}

// Output-side benchmark: consume one value and return 1--5 values.
at::Tensor tensor_out_1(const at::Tensor& x) { return x; }
std::tuple<at::Tensor, at::Tensor> tensor_out_2(const at::Tensor& x) { return {x, x}; }
std::tuple<at::Tensor, at::Tensor, at::Tensor> tensor_out_3(const at::Tensor& x) { return {x, x, x}; }
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> tensor_out_4(const at::Tensor& x) { return {x, x, x, x}; }
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> tensor_out_5(const at::Tensor& x) { return {x, x, x, x, x}; }

int64_t int_out_1(int64_t x) { return x; }
std::tuple<int64_t, int64_t> int_out_2(int64_t x) { return {x, x}; }
std::tuple<int64_t, int64_t, int64_t> int_out_3(int64_t x) { return {x, x, x}; }
std::tuple<int64_t, int64_t, int64_t, int64_t> int_out_4(int64_t x) { return {x, x, x, x}; }
std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t> int_out_5(int64_t x) { return {x, x, x, x, x}; }

TORCH_LIBRARY(pvd, m) {
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
TORCH_LIBRARY_IMPL(pvd, CompositeExplicitAutograd, m) {
  m.impl("my_op", TORCH_FN(my_op));
  m.impl("my_op_work", TORCH_FN(my_op_work));
  m.impl("tensor_in_1", TORCH_FN(tensor_in_1));
  m.impl("tensor_in_2", TORCH_FN(tensor_in_2));
  m.impl("tensor_in_3", TORCH_FN(tensor_in_3));
  m.impl("tensor_in_4", TORCH_FN(tensor_in_4));
  m.impl("tensor_in_5", TORCH_FN(tensor_in_5));
  m.impl("tensor_out_1", TORCH_FN(tensor_out_1));
  m.impl("tensor_out_2", TORCH_FN(tensor_out_2));
  m.impl("tensor_out_3", TORCH_FN(tensor_out_3));
  m.impl("tensor_out_4", TORCH_FN(tensor_out_4));
  m.impl("tensor_out_5", TORCH_FN(tensor_out_5));
}
TORCH_LIBRARY_IMPL(pvd, CatchAll, m) {
  m.impl("int_in_1", TORCH_FN(int_in_1));
  m.impl("int_in_2", TORCH_FN(int_in_2));
  m.impl("int_in_3", TORCH_FN(int_in_3));
  m.impl("int_in_4", TORCH_FN(int_in_4));
  m.impl("int_in_5", TORCH_FN(int_in_5));
  m.impl("int_out_1", TORCH_FN(int_out_1));
  m.impl("int_out_2", TORCH_FN(int_out_2));
  m.impl("int_out_3", TORCH_FN(int_out_3));
  m.impl("int_out_4", TORCH_FN(int_out_4));
  m.impl("int_out_5", TORCH_FN(int_out_5));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("my_op", &my_op);
  m.def("my_op_work", &my_op_work);
  m.def("tensor_in_1", &tensor_in_1);
  m.def("tensor_in_2", &tensor_in_2);
  m.def("tensor_in_3", &tensor_in_3);
  m.def("tensor_in_4", &tensor_in_4);
  m.def("tensor_in_5", &tensor_in_5);
  m.def("int_in_1", &int_in_1);
  m.def("int_in_2", &int_in_2);
  m.def("int_in_3", &int_in_3);
  m.def("int_in_4", &int_in_4);
  m.def("int_in_5", &int_in_5);
  m.def("tensor_out_1", &tensor_out_1);
  m.def("tensor_out_2", &tensor_out_2);
  m.def("tensor_out_3", &tensor_out_3);
  m.def("tensor_out_4", &tensor_out_4);
  m.def("tensor_out_5", &tensor_out_5);
  m.def("int_out_1", &int_out_1);
  m.def("int_out_2", &int_out_2);
  m.def("int_out_3", &int_out_3);
  m.def("int_out_4", &int_out_4);
  m.def("int_out_5", &int_out_5);
}
