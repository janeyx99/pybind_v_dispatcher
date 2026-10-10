"""Binding/dispatcher input and output overhead by type and arity.

Input-side kernels consume 1--5 values and return nothing. Output-side kernels
consume one value and return 1--5 values. The bodies do no other work.

    python setup.py build_ext --inplace
    python bench_args.py [min_run_time_s] [repeats]
"""

import random
import statistics
import sys

import torch
import torch.utils.benchmark as tb

import pvd


MIN_RUN_TIME = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
ARITIES = range(1, 6)
ARM_NAMES = (
    "pybind",
    "Python-registered pybind",
    "TORCH_LIBRARY",
    "STABLE_TORCH_LIBRARY",
)

# Set this once, before any kernels execute.  Timer otherwise restores the
# machine-wide default after every sample, which can warn after thread-pool use.
torch.set_num_threads(1)

# Register the existing pybind functions as Python dispatcher kernels. A call
# to this arm enters through torch.ops, dispatches to Python, and then invokes
# the pybind function.  CompositeExplicitAutograd matches the C++ arm's key.
PYBIND_LIBRARY = torch.library.Library("pvd_pybind", "DEF")


def operator_schema(kind, direction, count):
    name = f"{kind}_{direction}_{count}"
    if direction == "in":
        arg_type = "Tensor" if kind == "tensor" else "int"
        parameters = ", ".join(f"{arg_type} a{i}" for i in range(1, count + 1))
        returns = "()"
    else:
        if kind == "tensor":
            parameters = "Tensor(a) x"
            result_type = "Tensor(a)"
        else:
            parameters = "int x"
            result_type = "int"
        returns = result_type if count == 1 else "(" + ", ".join(
            [result_type] * count
        ) + ")"
    return name, f"{name}({parameters}) -> {returns}"


for argument_kind in ("tensor", "int"):
    for side in ("in", "out"):
        for arity in ARITIES:
            op_name, schema = operator_schema(argument_kind, side, arity)
            PYBIND_LIBRARY.define(schema)
            PYBIND_LIBRARY.impl(
                op_name, getattr(pvd, op_name), "CompositeExplicitAutograd"
            )


def stable_op(name):
    try:
        return getattr(torch.ops.pvd_stable, name)
    except AttributeError as exc:
        raise SystemExit(
            "STABLE_TORCH_LIBRARY operators are unavailable; use PyTorch >= 2.10 "
            "and rebuild the extension"
        ) from exc


def arms(kind, direction, count):
    name = f"{kind}_{direction}_{count}"
    return {
        "pybind": getattr(pvd, name),
        "Python-registered pybind": getattr(torch.ops.pvd_pybind, name),
        "TORCH_LIBRARY": getattr(torch.ops.pvd, name),
        "STABLE_TORCH_LIBRARY": stable_op(name),
    }


def arguments(kind, direction, count):
    argument_count = count if direction == "in" else 1
    if kind == "tensor":
        return tuple(torch.randn(8) for _ in range(argument_count))
    return tuple(range(1, argument_count + 1))


def measure(fn, args):
    for _ in range(1000):
        fn(*args)
    namespace = {"fn": fn}
    namespace.update({f"a{i}": value for i, value in enumerate(args, 1)})
    statement = "fn(" + ", ".join(f"a{i}" for i in range(1, len(args) + 1)) + ")"
    result = tb.Timer(statement, globals=namespace, num_threads=1).blocked_autorange(
        min_run_time=MIN_RUN_TIME
    )
    return result.median * 1e9


def validate(kind, direction, count, args, funcs):
    expected = args[0]
    for name, fn in funcs.items():
        actual = fn(*args)
        if direction == "in":
            valid = actual is None
        else:
            actual_values = (actual,) if count == 1 else actual
            valid = isinstance(actual_values, tuple) and len(actual_values) == count
            if valid and kind == "tensor":
                valid = all(torch.equal(value, expected) for value in actual_values)
            elif valid:
                valid = all(value == expected for value in actual_values)
        if not valid:
            raise SystemExit(f"{name} returned a different result")


def run_suite(kind, direction):
    samples = {
        arity: {name: [] for name in ARM_NAMES}
        for arity in ARITIES
    }
    for repeat in range(REPEATS):
        jobs = [(arity, name) for arity in ARITIES for name in ARM_NAMES]
        random.Random(20260928 + repeat).shuffle(jobs)
        cached = {}
        for arity, name in jobs:
            if arity not in cached:
                args = arguments(kind, direction, arity)
                funcs = arms(kind, direction, arity)
                validate(kind, direction, arity, args, funcs)
                cached[arity] = args, funcs
            args, funcs = cached[arity]
            samples[arity][name].append(measure(funcs[name], args))
    return {
        arity: {
            name: (
                statistics.median(samples[arity][name]),
                min(samples[arity][name]),
                max(samples[arity][name]),
            )
            for name in ARM_NAMES
        }
        for arity in ARITIES
    }


def print_table(kind, direction, results):
    if direction == "in":
        title = f"{kind.upper()} INPUTS (1-5 inputs, no outputs)"
        count_label = "inputs"
    else:
        title = f"{kind.upper()} OUTPUTS (one input, 1-5 outputs)"
        count_label = "outputs"
    print(f"\n{title}; median [min-max] ns/call across {REPEATS} runs")
    print(f"  {count_label:>7}" + "".join(f"{name:>27}" for name in ARM_NAMES))
    for arity in ARITIES:
        cells = []
        for name in ARM_NAMES:
            median, low, high = results[arity][name]
            cells.append(f"{median:.0f} [{low:.0f}-{high:.0f}]")
        print(
            f"  {arity:>7}"
            + "".join(f"{cell:>27}" for cell in cells)
        )


print(
    f"torch {torch.__version__} | blocked_autorange(min_run_time={MIN_RUN_TIME}s) "
    f"| repeats={REPEATS} | num_threads=1"
)
print("all results are validated; measurement order is shuffled")
for benchmark_side in ("in", "out"):
    for argument_kind in ("tensor", "int"):
        print_table(
            argument_kind,
            benchmark_side,
            run_suite(argument_kind, benchmark_side),
        )
