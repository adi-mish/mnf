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

    feature_baselines = data.get("feature_baseline_suite")
    if feature_baselines is not None:
        lines.extend([
            "## Feature Baselines",
            "",
            f"Mean false-mechanism rate on random labelable controls: `{_fmt(feature_baselines['mean_random_false_mechanism_rate'])}`.",
            f"Mean causal-use score: random labelable `{_fmt(feature_baselines['mean_random_causal_use_score'])}` versus trained-used `{_fmt(feature_baselines['mean_trained_causal_use_score'])}`.",
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
        context_stability = interaction.get("context_stability")
        recovery = interaction.get("interaction_recovery")
        higher_order = interaction.get("higher_order")
        noisy = interaction.get("noisy_interaction_recovery")
        active = interaction.get("active_design")
        active_baselines = interaction.get("active_design_baselines")
        lines.extend([
            "## Mechanism Interactions",
            "",
            f"Redundant-path dual ablation drop: `{_fmt(redundant['dual_ablation_drop'])}` with single-ablation drops `{_fmt(redundant['drop_m_from_joint'])}` and `{_fmt(redundant['drop_n_from_joint'])}`.",
            f"Gating strength `gate -> worker`: `{_fmt(gating['gate_m_to_n'])}`.",
            f"Shared-MDL gain for reused route atom: `{_fmt(shared['shared_mdl_gain'])}`.",
            f"Maximum capacity-competition score in the sweep: `{_fmt(interaction['max_capacity_competition'])}`.",
            f"Factorial interaction phase-diagram rows: `{interaction['phase_diagram_rows']}`.",
        ])
        if recovery is not None:
            lines.append(
                f"Interaction recovery F1: `{_fmt(recovery['recovery']['f1'])}` over `{recovery['n_pairs']}` mechanism pairs using `{recovery['design_size']}` intervention states."
            )
        table_aliasing = interaction.get("table_aliasing")
        if table_aliasing is not None:
            lines.append(
                f"Structural aliasing: behavior-only label `{table_aliasing['behavior_only']['structural_label']}`; internal evidence orients the directed gate as `{table_aliasing['directed_structural']['orientation']}`."
            )
        if context_stability is not None:
            lines.append(
                f"Context-stability changed pairs: `{context_stability['n_changed_pairs']}` of `{context_stability['n_pairs']}`; the focal pair changes from `{context_stability['focal_context_on_label']}` to `{context_stability['focal_context_off_label']}`."
            )
        context_sweep = interaction.get("context_stability_sweep")
        if context_sweep is not None:
            lines.append(
                f"Context-stability sweep changed-context rate: `{_fmt(context_sweep['changed_context_rate'])}` over `{context_sweep['n_rows']}` parameter settings."
            )
        if higher_order is not None:
            lines.append(
                f"Higher-order triple-gate contrast: `{_fmt(higher_order['third_order_effect'])}` using `{higher_order['triple_design_size']}` intervention states."
            )
            lines.append(
                f"Pairwise-only higher-order search labels: `{higher_order['pairwise_only_search']['label_counts']}`; order-3 search labels: `{higher_order['triple_search']['label_counts']}`."
            )
        if noisy is not None:
            low_noise = noisy["rows"][1] if len(noisy["rows"]) > 1 else noisy["rows"][0]
            high_noise = noisy["rows"][-1]
            lines.append(
                f"Noisy recovery all-correct rate: `{_fmt(low_noise['all_correct_rate'])}` at noise `{low_noise['noise']}` and `{_fmt(high_noise['all_correct_rate'])}` at noise `{high_noise['noise']}`."
            )
            lines.append(
                f"Noisy recovery abstention rate: `{_fmt(low_noise['mean_uncertain_rate'])}` at noise `{low_noise['noise']}` and `{_fmt(high_noise['mean_uncertain_rate'])}` at noise `{high_noise['noise']}`."
            )
        if active is not None:
            clean_active = active["rows"][0]
            last_active = active["rows"][-1]
            lines.append(
                f"Active design mean measurements: `{_fmt(clean_active['mean_measurements'])}` at noise `{clean_active['noise']}` and `{_fmt(last_active['mean_measurements'])}` at noise `{last_active['noise']}` with accuracy `{_fmt(last_active['mean_accuracy'])}`."
            )
        if active_baselines is not None:
            baseline_row = next(
                row
                for row in active_baselines["rows"]
                if row["noise"] == 0.02 and row["budget"] == 64
            )
            lines.append(
                f"Active-vs-baseline at noise `0.02`, budget `64`: active stable rate `{_fmt(baseline_row['active_stable_rate'])}`, uniform `{_fmt(baseline_row['uniform_stable_rate'])}`, random `{_fmt(baseline_row['random_stable_rate'])}`."
            )
        lines.append("")

    atlas = data.get("atlas_suite")
    if atlas is not None:
        no_global = atlas["no_global_chart"]
        lines.extend([
            "## Atlas Gluing",
            "",
            f"Best global glue error: `{_fmt(no_global['best_global_glue_error'])}` versus context-indexed glue error `{_fmt(no_global['context_indexed_glue_error'])}`.",
            "",
        ])

    certificate = data.get("certificate_demo")
    if certificate is not None:
        report = certificate["report"]
        faithful = report["certificates"].get("faithful_but_long", {})
        identification = faithful.get("identification", {})
        lines.extend([
            "## Certificate Vectors",
            "",
            f"Pareto frontier indices: `{report['pareto_indices']}`.",
            f"Accepted certificates: `{report['accepted']}`.",
        ])
        if identification:
            lines.append(
                f"Example identification set: `{identification['representatives']}` with diameter `{_fmt(identification['diameter'])}`."
            )
        lines.append("")

    identifiability = data.get("identifiability_demo")
    if identifiability is not None:
        atom = identifiability["atom_splitting"]
        output_only = atom["output_only_identification"]
        representatives = output_only["representatives"]
        lines.extend([
            "## Identifiability",
            "",
            f"Output-only atom-splitting max distance: `{_fmt(atom['output_only_max_distance'])}` with representatives `{representatives}` and ID diameter `{_fmt(output_only['diameter'])}`.",
            f"Adding internal route markers raises max distance to `{_fmt(atom['rich_observable_max_distance'])}`.",
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

    tiny = data.get("tiny_transformer_demo")
    if tiny is not None:
        lines.extend([
            "## Tiny Transformer",
            "",
        ])
        if tiny.get("available"):
            site_summary = tiny.get("mean_activation_site_patch_consistency", {})
            wrong_site_summary = tiny.get("mean_activation_site_wrong_token_consistency", {})
            lines.extend([
                f"Mean final modular-addition accuracy: `{_fmt(tiny['mean_final_accuracy'])}`.",
                f"Mean cyclic-shift consistency: input-a `{_fmt(tiny['mean_a_cyclic_shift_consistency'])}`, input-b `{_fmt(tiny['mean_b_cyclic_shift_consistency'])}`.",
                f"Mean embedding-patch consistency: input-a `{_fmt(tiny['mean_a_embedding_patch_consistency'])}`, input-b `{_fmt(tiny['mean_b_embedding_patch_consistency'])}`.",
            ])
            if site_summary:
                formatted_sites = ", ".join(f"{site}: `{_fmt(value)}`" for site, value in site_summary.items())
                lines.append(f"Mean deeper activation-site patch consistency: {formatted_sites}.")
            if wrong_site_summary:
                formatted_wrong_sites = ", ".join(
                    f"{site}: `{_fmt(value)}`" for site, value in wrong_site_summary.items()
                )
                lines.append(f"Mean wrong-token patch consistency controls: {formatted_wrong_sites}.")
            lines.append("")
        else:
            lines.extend([
                f"Skipped: `{tiny['reason']}`.",
                "",
            ])

    redundant_tiny = data.get("tiny_redundant_transformer_demo")
    if redundant_tiny is not None:
        lines.extend([
            "## Tiny Redundant Transformer",
            "",
        ])
        if redundant_tiny.get("available"):
            lines.extend([
                f"Mean base accuracy: `{_fmt(redundant_tiny['mean_base_accuracy'])}`.",
                f"Mean route-only accuracy: route-a `{_fmt(redundant_tiny['mean_route_a_only_accuracy'])}`, route-b `{_fmt(redundant_tiny['mean_route_b_only_accuracy'])}`.",
                f"Mean both-routes-ablated accuracy: `{_fmt(redundant_tiny['mean_both_routes_ablated_accuracy'])}`.",
                f"Mean max single-ablation drop: `{_fmt(redundant_tiny['mean_max_single_ablation_drop'])}` versus dual-ablation drop `{_fmt(redundant_tiny['mean_dual_ablation_drop'])}`.",
                f"Redundancy certified rate: `{_fmt(redundant_tiny['redundancy_certified_rate'])}`; single-ablation underweights rate: `{_fmt(redundant_tiny['single_ablation_underweights_rate'])}`.",
                "",
            ])
        else:
            lines.extend([
                f"Skipped: `{redundant_tiny['reason']}`.",
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
