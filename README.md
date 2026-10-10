# pybind_v_dispatcher

Moving a PyTorch op to the stable ABI means registering it with the dispatcher, so leaving
`pybind11` is mandatory on that path — this measures what that step costs. The same operation
is exposed three ways (`PYBIND11_MODULE`, `TORCH_LIBRARY`, `STABLE_TORCH_LIBRARY`); the first
two share a C++ symbol, while the stable arm has an equivalent stable-ABI implementation.
Timing is `torch.utils.benchmark.blocked_autorange`
with `num_threads=1`; the arms are checked for identical results before anything is timed.
For a fixed schema the toll is device-independent — measured at +473 ns on CPU and +476 ns
on CUDA tensors, a difference well inside the IQR. It is not independent of the schema:
`bench_args.py` separately measures one to five tensor or integer inputs and outputs.

What the toll is a fraction of depends entirely on the device. A CPU op's cost grows with
`numel`, so the toll decays to ~0.1%; a dispatched CUDA op is launch-bound at ~5.5 µs of
host time regardless of size, so the toll sits at a flat ~8% until the kernel is large
enough to backpressure the launch queue. That is why a decode-shaped GPU workload pays the
toll on every eager call, per layer per token, while a training-shaped one does not notice
it. These are eager-mode numbers; a captured CUDA graph replays launches without the
dispatcher and does not pay it at all.
It is also not pure overhead: a `pybind11` call fails `torch.compile(fullgraph=True)` and
emits no profiler event, so the dispatcher charges for work the pybind arm never does.
To measure your own op, replace the marked block in `bindings.cpp` (and `stable.cpp` for
arm C) and run it a few times — between-run drift exceeded the within-run IQR on our box.

## Run

```bash
python setup.py build_ext --inplace
python bench.py                 # or: python bench.py 5.0
python bench_args.py            # optional: python bench_args.py 1.0 3
```

`bench_args.py` separately measures 1--5 inputs with no output and 1--5 outputs with
one input, for both tensors and integers. It compares direct pybind, that same pybind
function registered as a Python dispatcher kernel, `TORCH_LIBRARY` through `torch.ops`,
and `STABLE_TORCH_LIBRARY` through `torch.ops`. Inputs are created outside the timed
region, and fixed-arity statements avoid accidentally timing Python `*args` unpacking.

### Tensor arity

The no-op input kernels in [`bindings.cpp`](bindings.cpp#L21-L32) consume 1--5 values
and return nothing. The output kernels in [`bindings.cpp`](bindings.cpp#L34-L45) take
one value and return 1--5 aliases, so the timings exclude tensor allocation, copying,
and computation. The stable arm implements the same operations with stable-ABI types.

Median ns/call across three shuffled runs on PyTorch 2.15.0a0+gitb1dc65e:

| Binding | Tensor inputs, 1 → 5 | Tensor outputs, 1 → 5 |
|---|---:|---:|
| pybind11 | 89  →  303 | 96  →  135 |
| pybind11 + Dispatcher | 1,003  →  1,638 | 1,016  →  1,335 |
| TORCH_LIBRARY | 600  →  1,108 | 591  →  750 |
| STABLE_TORCH_LIBRARY | 672  →  1,322 | 682  →  1,043 |

All paths scale with arity. Relative to pybind11, `TORCH_LIBRARY` adds about 74 ns per
additional tensor input and 30 ns per additional tensor output; `STABLE_TORCH_LIBRARY`
adds about 109 ns and 81 ns, respectively. Moving from `TORCH_LIBRARY` to
`STABLE_TORCH_LIBRARY` costs 72 ns for one tensor input and 91 ns for one tensor output,
then about 36 ns per additional input and 51 ns per additional output. The
Python-registered pybind arm is slowest because it crosses the dispatcher into Python
and then invokes pybind11.

```
torch 2.15.0a0+gitc312cf8 | blocked_autorange(min_run_time=2.0s) | num_threads=1
all arms return identical results

THE TOLL (identity body: pure call overhead)
  arm                              no grad      autograd   ns/call (IQR)
  A  pybind11                      94 (  2)       94 (  2)
  B  TORCH_LIBRARY                562 ( 15)     1089 ( 23)
  C  STABLE_TORCH_LIBRARY         670 ( 50)     1274 ( 50)

  pybind11 -> TORCH_LIBRARY           +468          +995   mandatory for the stable ABI
  TORCH_LIBRARY -> STABLE             +108          +184

IN CONTEXT (the toll is fixed; what it is a fraction of is not)
  device      numel   op host ns   toll as % of the call
  cpu          1024         2336                  16.68%
  cpu       1048576       291727                   0.16%
  cuda         1024         5999                   7.23%
  cuda      1048576         5855                   7.40%
  cuda     16777216        47708                   0.97%

  On CUDA the host cost is launch-bound and flat, so the toll does not decay
  with tensor size -- it stays ~8% until the kernel backpressures the queue.

WHAT IT BUYS
  arm                         torch.compile  profiler
  A  pybind11                   Unsupported      none
  B  TORCH_LIBRARY             fullgraph ok       yes
  C  STABLE_TORCH_LIBRARY      fullgraph ok       yes

Run this a few times: between-run drift exceeded the within-run IQR on our box.
```

## License

MIT — see [LICENSE](LICENSE).
