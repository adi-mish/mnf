from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mnf.integrations.tracr.availability import tracr_status


@dataclass(frozen=True)
class TracrProgramSmoke:
    name: str
    available: bool
    decoded: tuple[object, ...] = ()
    expected: tuple[object, ...] = ()
    exact_match: bool = False
    residual_shapes: tuple[tuple[int, ...], ...] = ()
    layer_output_shapes: tuple[tuple[int, ...], ...] = ()
    attn_logit_shapes: tuple[tuple[int, ...], ...] = ()
    transformer_output_shape: tuple[int, ...] = ()
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "available": self.available,
            "decoded": list(self.decoded),
            "expected": list(self.expected),
            "exact_match": self.exact_match,
            "residual_shapes": [list(shape) for shape in self.residual_shapes],
            "layer_output_shapes": [list(shape) for shape in self.layer_output_shapes],
            "attn_logit_shapes": [list(shape) for shape in self.attn_logit_shapes],
            "transformer_output_shape": list(self.transformer_output_shape),
            "reason": self.reason,
        }


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value.shape)


def _matches(decoded: Sequence[object], expected: Sequence[object], atol: float = 1e-6) -> bool:
    if len(decoded) != len(expected):
        return False
    for got, want in zip(decoded, expected):
        if want is None:
            continue
        if isinstance(want, (int, float)) and isinstance(got, (int, float, np.number)):
            if abs(float(got) - float(want)) > atol:
                return False
        elif got != want:
            return False
    return True


def _program_specs() -> dict[str, dict[str, object]]:
    from tracr.compiler import lib
    from tracr.rasp import rasp

    count_a = lib.make_count(rasp.tokens, "a")
    count_a_copy = rasp.Map(lambda c: c, count_a)
    reverse_tokens = lib.make_reverse(rasp.tokens)
    increment_mod_five = rasp.Map(lambda x: (x + 1) % 5, rasp.tokens)
    count_zero = lib.make_count(rasp.tokens, 0)

    return {
        "reverse": {
            "program": lib.make_reverse(rasp.tokens),
            "vocab": {"a", "b", "c", "d"},
            "input": list("abcd"),
            "expected": list("dcba"),
            "max_seq_len": 5,
        },
        "histogram": {
            "program": lib.make_hist(),
            "vocab": {"a", "b", "c"},
            "input": list("abaca"),
            "expected": [3, 1, 3, 1, 3],
            "max_seq_len": 6,
        },
        "token_frequency": {
            "program": lib.make_count(rasp.tokens, "a"),
            "vocab": {"a", "b", "c"},
            "input": list("abaca"),
            "expected": [3, 3, 3, 3, 3],
            "max_seq_len": 6,
        },
        "sort": {
            "program": lib.make_sort(rasp.tokens, rasp.tokens, max_seq_len=5, min_key=0),
            "vocab": {1, 2, 3, 4},
            "input": [3, 1, 4, 2],
            "expected": [1, 2, 3, 4],
            "max_seq_len": 5,
        },
        "balanced_parentheses": {
            "program": lib.make_shuffle_dyck(["()"]),
            "vocab": {"(", ")"},
            "input": list("(())"),
            "expected": [True, True, True, True],
            "max_seq_len": 5,
        },
        "modular_arithmetic": {
            "program": rasp.Map(lambda x: (x + 1) % 5, rasp.tokens),
            "vocab": {0, 1, 2, 3, 4},
            "input": [0, 1, 4],
            "expected": [1, 2, 0],
            "max_seq_len": 4,
        },
        "shared_subroutine": {
            "program": rasp.SequenceMap(lambda hist, count: hist + count, lib.make_hist(), count_a),
            "vocab": {"a", "b", "c"},
            "input": list("abaca"),
            "expected": [6, 4, 6, 4, 6],
            "max_seq_len": 6,
        },
        "redundant_subroutines": {
            "program": rasp.SequenceMap(lambda left, right: max(left, right), count_a, count_a_copy),
            "vocab": {"a", "b", "c"},
            "input": list("abaca"),
            "expected": [3, 3, 3, 3, 3],
            "max_seq_len": 6,
        },
        "context_gated_subroutine": {
            "program": rasp.SequenceMap(lambda token, count: count if token == "a" else 0, rasp.tokens, count_a),
            "vocab": {"a", "b", "c"},
            "input": list("abaca"),
            "expected": [3, 0, 3, 0, 3],
            "max_seq_len": 6,
        },
        "shortcut_plus_correct_algorithm": {
            "program": rasp.SequenceMap(
                lambda token, reversed_token: token if token == "s" else reversed_token,
                rasp.tokens,
                reverse_tokens,
            ),
            "vocab": {"a", "b", "c", "s"},
            "input": list("abcs"),
            "expected": ["s", "c", "b", "s"],
            "max_seq_len": 5,
        },
        "cyclic_arithmetic_plus_lookup": {
            "program": rasp.SequenceMap(lambda inc, count: (inc + count) % 5, increment_mod_five, count_zero),
            "vocab": {0, 1, 2, 3, 4},
            "input": [0, 1, 4],
            "expected": [2, 3, 1],
            "max_seq_len": 4,
        },
        "length": {
            "program": rasp.categorical(lib.make_length()),
            "vocab": {"a", "b", "c", "d"},
            "input": list("abc"),
            "expected": [3, 3, 3],
            "max_seq_len": 5,
        },
        "map_increment": {
            "program": rasp.Map(lambda x: x + 1, rasp.tokens),
            "vocab": {0, 1, 2},
            "input": [0, 1, 2],
            "expected": [1, 2, 3],
            "max_seq_len": 4,
        },
    }


def run_tracr_program_smoke(name: str, compiler_bos: str = "BOS") -> TracrProgramSmoke:
    status = tracr_status()
    if not status.available:
        return TracrProgramSmoke(name=name, available=False, reason=status.reason)
    try:
        from tracr.compiler import compiling

        specs = _program_specs()
        if name not in specs:
            raise ValueError(f"unknown Tracr smoke program: {name}")
        spec = specs[name]
        assembled = compiling.compile_rasp_to_model(
            spec["program"],
            spec["vocab"],
            int(spec["max_seq_len"]),
            compiler_bos=compiler_bos,
        )
        output = assembled.apply([compiler_bos] + list(spec["input"]))
        decoded = tuple(output.decoded[1:])
        expected = tuple(spec["expected"])
        return TracrProgramSmoke(
            name=name,
            available=True,
            decoded=decoded,
            expected=expected,
            exact_match=_matches(decoded, expected),
            residual_shapes=tuple(_shape(value) for value in output.residuals),
            layer_output_shapes=tuple(_shape(value) for value in output.layer_outputs),
            attn_logit_shapes=tuple(_shape(value) for value in output.attn_logits),
            transformer_output_shape=_shape(output.transformer_output),
            reason="ok",
        )
    except Exception as exc:
        return TracrProgramSmoke(name=name, available=False, reason=repr(exc))


def run_tracr_smoke(
    programs: Sequence[str] = (
        "reverse",
        "histogram",
        "token_frequency",
        "sort",
        "balanced_parentheses",
        "modular_arithmetic",
        "shared_subroutine",
        "redundant_subroutines",
        "context_gated_subroutine",
        "shortcut_plus_correct_algorithm",
        "cyclic_arithmetic_plus_lookup",
    ),
) -> dict[str, object]:
    rows = [run_tracr_program_smoke(name).as_dict() for name in programs]
    return {
        "available": bool(rows and all(row["available"] for row in rows)),
        "n_programs": len(rows),
        "all_exact_match": bool(rows and all(row["exact_match"] for row in rows)),
        "rows": rows,
    }
