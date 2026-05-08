from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence
import numpy as np

from mnf.core.causal_program import CausalProgram
from mnf.core.interventions import Intervention
from mnf.core.mdl import object_description_length, program_description_length
from mnf.core.metrics import intervention_prediction_error, mse


Encoder = Callable[[Any], Mapping[str, Any]]
Decoder = Callable[[Mapping[str, Any]], Any]
InterventionMap = Callable[[Intervention], Intervention]


@dataclass(frozen=True)
class MNFScore:
    total: float
    obs_error: float
    intervention_error: float
    invariance_error: float
    description_length: float

    def as_dict(self) -> dict[str, float]:
        return {
            "total": self.total,
            "obs_error": self.obs_error,
            "intervention_error": self.intervention_error,
            "invariance_error": self.invariance_error,
            "description_length": self.description_length,
        }


@dataclass
class MechanismExplanation:
    """Candidate MNF explanation for a model domain.

    model: callable mapping inputs -> low-level outputs or activations.
    program: high-level causal program.
    encoder: low-level output/activation -> high-level state.
    decoder: high-level state -> low-level output prediction.
    intervention_map: high-level intervention -> model-level intervention.  In
      this scaffold the model intervention is optional; synthetic evaluators use
      known model-intervention callables.
    """

    program: CausalProgram
    encoder: Encoder
    decoder: Decoder
    name: str = "mechanism"
    parameters_description: Any = None

    def predict_low_level(self, inputs: Mapping[str, Any]) -> Any:
        state = self.program.run(inputs)
        return self.decoder(state)

    def obs_error(
        self,
        inputs: Sequence[Mapping[str, Any]],
        low_level_outputs: Sequence[Any],
        loss_fn: Callable[[Any, Any], float] = mse,
    ) -> float:
        if len(inputs) != len(low_level_outputs):
            raise ValueError("inputs and outputs length mismatch")
        if not inputs:
            return 0.0
        return float(np.mean([loss_fn(y, self.predict_low_level(x)) for x, y in zip(inputs, low_level_outputs)]))

    def intervention_error(
        self,
        inputs: Sequence[Mapping[str, Any]],
        interventions: Sequence[Mapping[str, Intervention]],
        observed_states: Sequence[Mapping[str, Any]],
        keys: Sequence[str] | None = None,
    ) -> float:
        predicted = [self.program.run(x, intervention) for x, intervention in zip(inputs, interventions)]
        return intervention_prediction_error(observed_states, predicted, keys=keys)

    def invariance_error(
        self,
        paired_low_level_states: Sequence[tuple[Any, Any]],
        keys: Sequence[str] | None = None,
    ) -> float:
        """Distance between encoded states for paraphrase/checkpoint pairs."""
        if len(paired_low_level_states) == 0:
            return 0.0
        errs = []
        for a, b in paired_low_level_states:
            za = self.encoder(a)
            zb = self.encoder(b)
            use_keys = keys or tuple(sorted(set(za) & set(zb)))
            for key in use_keys:
                va, vb = za[key], zb[key]
                if isinstance(va, (int, float, np.number)) and isinstance(vb, (int, float, np.number)):
                    errs.append((float(va) - float(vb)) ** 2)
                else:
                    errs.append(0.0 if va == vb else 1.0)
        return float(np.mean(errs)) if errs else 0.0

    def description_length(self) -> float:
        return program_description_length(self.program) + object_description_length(self.parameters_description)

    def score(
        self,
        inputs: Sequence[Mapping[str, Any]],
        low_level_outputs: Sequence[Any],
        interventions: Sequence[Mapping[str, Intervention]] | None = None,
        observed_intervention_states: Sequence[Mapping[str, Any]] | None = None,
        paired_low_level_states: Sequence[tuple[Any, Any]] | None = None,
        lambda_int: float = 1.0,
        mu_inv: float = 1.0,
        beta_mdl: float = 0.01,
    ) -> MNFScore:
        obs = self.obs_error(inputs, low_level_outputs)
        if interventions is not None and observed_intervention_states is not None:
            int_err = self.intervention_error(inputs, interventions, observed_intervention_states)
        else:
            int_err = 0.0
        inv_err = self.invariance_error(paired_low_level_states or [])
        mdl = self.description_length()
        total = obs + lambda_int * int_err + mu_inv * inv_err + beta_mdl * mdl
        return MNFScore(total, obs, int_err, inv_err, mdl)
