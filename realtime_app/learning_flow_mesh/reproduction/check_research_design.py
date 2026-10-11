"""Analytic counterexamples for the next research design; no learned model.

These fixtures test mathematical failure modes, not human reconstruction accuracy.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def run_checks():
    results = {}
    angle = np.pi / 3
    bones = np.array([[np.cos(angle), np.sin(angle)],
                      [np.cos(angle), -np.sin(angle)]])
    lengths = np.linalg.norm(bones, axis=1)
    averaged_length = np.linalg.norm(bones.mean(axis=0))
    assert np.allclose(lengths, 1) and np.isclose(averaged_length, 0.5)
    results["coordinate_mean_shortens_bone"] = {
        "input_lengths": lengths.tolist(), "mean_length": float(averaged_length)}

    rotation_a = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    rotation_b = rotation_a.T
    rotation_mean = (rotation_a + rotation_b) / 2
    assert np.isclose(np.linalg.det(rotation_a), 1)
    assert np.isclose(np.linalg.det(rotation_mean), 0)
    results["rotation_matrix_mean_is_not_rotation"] = {
        "mean_determinant": float(np.linalg.det(rotation_mean))}

    # Prior N(0,1), observation y=1 with N(0,1) noise.
    # The correct conditional is N(1/2,1/2). Reusing the likelihood gives N(2/3,1/3).
    posterior_precision, posterior_information = 2., 1.
    repeated_precision, repeated_information = posterior_precision + 1, posterior_information + 1
    results["repeated_observation_changes_posterior"] = {
        "correct_mean": posterior_information / posterior_precision,
        "correct_variance": 1 / posterior_precision,
        "reweighted_mean": repeated_information / repeated_precision,
        "reweighted_variance": 1 / repeated_precision}
    assert results["repeated_observation_changes_posterior"]["reweighted_variance"] < 0.5

    # y = beta + pose: beta has a nonzero derivative but cannot be isolated.
    jacobian = np.array([[1., 1.]])
    information = jacobian.T @ jacobian
    marginal_shape_information = information[0, 0] - (
        information[0, 1] * np.linalg.pinv(information[1:, 1:])[0, 0] * information[1, 0])
    assert information[0, 0] == 1 and marginal_shape_information == 0
    results["shape_pose_confounding"] = {
        "naive_shape_information": float(information[0, 0]),
        "pose_eliminated_shape_information": float(marginal_shape_information)}

    n, correlation = 8, 0.9
    effective_count = n / (1 + (n - 1) * correlation)
    assert effective_count < 1.1
    results["correlated_frames_are_not_independent"] = {
        "count": n, "equicorrelation_assumption": correlation,
        "effective_count_for_mean": effective_count}

    # An equal two-point distribution versus its collapsed mean, evaluated at y=1.
    samples = np.array([-1., 1.])
    energy = np.abs(samples - 1).mean() - 0.5 * np.abs(samples[:, None] - samples[None]).mean()
    collapsed_energy = 1.
    assert energy == 0.5 and collapsed_energy > energy
    results["distribution_and_point_losses_differ"] = {
        "two_point_distribution_energy_score": float(energy),
        "collapsed_mean_energy_score": collapsed_energy,
        "mean_prediction_squared_error": 1., "random_sample_expected_squared_error": 2.}

    # A decision restricted to valid candidates cannot invent their midpoint.
    distances = np.linalg.norm(bones[:, None] - bones[None], axis=-1)
    medoid_index = int(np.argmin(distances.mean(axis=1)))
    assert np.isclose(np.linalg.norm(bones[medoid_index]), 1)
    results["candidate_medoid_preserves_fixture_bone"] = {
        "selected_index": medoid_index, "selected_length": float(np.linalg.norm(bones[medoid_index])),
        "limitation": "Membership does not prove the candidate is anatomically correct or accurate."}

    # A noise sample's ID is not a motion hypothesis ID across changed conditions.
    # In 2D, -I is the endpoint of a pi-angle rotational ODE (determinant +1).
    # Unlike a 1D reflection, both maps are realizable by continuous flows.
    noise = np.array([[-1., 0.], [1., 0.]])
    first, second = noise, -noise
    assert np.array_equal(first, second[::-1]) and not np.array_equal(first, second)
    results["shared_noise_does_not_identify_temporal_mode"] = {
        "first_outputs": first.tolist(), "second_outputs": second.tolist(),
        "same_marginal_different_correspondence": True,
        "construction": "Identity versus 2D pi rotation, both orientation-preserving ODE maps."}
    return {"status": "passed", "checks": results, "count": len(results),
            "boundary": "Analytic fixtures only; no training, checkpoint, body asset or real accuracy evaluation."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = run_checks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "count": report["count"]}))


if __name__ == "__main__":
    main()
