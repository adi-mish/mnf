from __future__ import annotations

from typing import Any, Mapping


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def summarize_research_sweeps(data: Mapping[str, Any]) -> str:
    """Render a compact Markdown summary of generated research sweeps."""

    lines = [
        "# Research sweep summary",
        "",
        "This file is generated from `docs/research_sweeps.json`.",
        "",
    ]

    gt = data["ground_truth_suite"]["rows"]
    lines.extend([
        "## Ground Truth Recovery",
        "",
        "| Benchmark | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: |",
    ])
    for row in gt:
        lines.append(
            f"| {row['benchmark']} | {_fmt(row['precision'])} | {_fmt(row['recall'])} | {_fmt(row['f1'])} |"
        )
    lines.append("")

    baseline = data["baseline_comparison"]["shortcut_selector_comparison"]
    train_only = baseline["train_only_selector"]
    invariant = baseline["invariance_selector"]
    lines.extend([
        "## Shortcut Baseline",
        "",
        "| Selector | Selected | Train Accuracy | Shifted Accuracy | Invariance Gap |",
        "| --- | --- | ---: | ---: | ---: |",
        f"| train-only | {train_only['selected']} | {_fmt(train_only['train_accuracy'])} | {_fmt(train_only['shifted_accuracy'])} | {_fmt(train_only['invariance_gap'])} |",
        f"| invariance | {invariant['selected']} | {_fmt(invariant['train_accuracy'])} | {_fmt(invariant['shifted_accuracy'])} | {_fmt(invariant['invariance_gap'])} |",
        "",
    ])

    cyclic_rows = data["cyclic_baseline_comparison"]["rows"]
    low_noise = cyclic_rows[0]
    high_noise = cyclic_rows[-1]
    lines.extend([
        "## Cyclic Baseline",
        "",
        f"At noise `{low_noise['noise']}`, typed rotation error is `{_fmt(low_noise['typed_rotation_error'])}` versus scalar rotation error `{_fmt(low_noise['scalar_rotation_error'])}`.",
        f"At noise `{high_noise['noise']}`, typed rotation error is `{_fmt(high_noise['typed_rotation_error'])}` versus scalar rotation error `{_fmt(high_noise['scalar_rotation_error'])}`.",
        "",
    ])

    induction = data["induction_demo"]
    lines.extend([
        "## Induction Match-Copy",
        "",
        f"Mean held-out memorizer accuracy: `{_fmt(induction['mean_memorizer_test_accuracy'])}`.",
        f"Mean held-out copy-mechanism accuracy: `{_fmt(induction['mean_copy_test_accuracy'])}`.",
        "",
    ])

    interaction = data.get("interaction_suite")
    if interaction is not None:
        redundant = interaction["redundant_paths"]
        gating = interaction["gating"]
        shared = interaction["shared_atom_reuse"]
        lines.extend([
            "## Mechanism Interactions",
            "",
            f"Redundant-path dual ablation drop: `{_fmt(redundant['dual_ablation_drop'])}` with single-ablation drops `{_fmt(redundant['drop_m_from_joint'])}` and `{_fmt(redundant['drop_n_from_joint'])}`.",
            f"Gating strength `gate -> worker`: `{_fmt(gating['gate_m_to_n'])}`.",
            f"Shared-MDL gain for reused route atom: `{_fmt(shared['shared_mdl_gain'])}`.",
            f"Maximum capacity-competition score in the sweep: `{_fmt(interaction['max_capacity_competition'])}`.",
            f"Factorial interaction phase-diagram rows: `{interaction['phase_diagram_rows']}`.",
            "",
        ])

    transition_rows = data["transition_atom_sweep"]["rows"]
    best_transition = min(transition_rows, key=lambda row: row["molt_to_global_mse_ratio"])
    lines.extend([
        "## Transition Atoms",
        "",
        f"Best MOLT/global MSE ratio: `{_fmt(best_transition['molt_to_global_mse_ratio'])}` at gate separation `{best_transition['gate_separation']}` and noise `{best_transition['noise']}`.",
        "",
    ])

    training = data["training_emergence"]
    lines.extend([
        "## Training Emergence",
        "",
        f"Mean mechanism lead: `{_fmt(training['mean_mechanism_lead_steps'])}` steps.",
        f"Mean mechanism/behavior correlation: `{_fmt(training['mean_correlation'])}`.",
        "",
    ])

    return "\n".join(lines)
