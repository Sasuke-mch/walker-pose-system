import pytest
import torch

from ..evaluate import candidate_metrics, energy_score, online_metrics, paired_bootstrap


def test_energy_multimodal_and_invalid_raw_not_silently_filtered():
    assert energy_score(torch.tensor([[-1.], [1.]]), torch.tensor([1.])) == .5
    points = torch.tensor([[[0., 0., 2.], [1., 0., 2.]], [[0., 0., 4.], [1., 0., 4.]],
                           [[torch.nan, 0., 0.], [0., 0., 0.]]])
    report = candidate_metrics(points, points[0], torch.tensor([True, True, False]),
                               torch.tensor([True, True, False]), selected_index=1)
    assert report["raw"]["energy_score"] is None
    assert report["numerically_legal"]["count"] == 2
    assert report["absolute_mpjpe_m_selected"] == 2
    assert report["root_relative_mpjpe_m_selected"] == 0
    assert report["selected_minus_oracle_m"] == 2


def test_subject_block_bootstrap_not_frame_pseudoreplication():
    records = [{"subject": subject, "sequence": "q", "block": block, "training_seed": seed,
                "baseline": 2., "proposed": 1.} for subject in ("a", "b") for block in range(2)
               for seed in (7, 17, 27)]
    result = paired_bootstrap(records, iterations=100)
    assert result["subjects"] == 2 and result["ci95"] == [-1., -1.]
    with pytest.raises(ValueError, match="seed coverage"):
        paired_bootstrap(records[:-1])


def test_missing_time_recovery_and_no_noise_id_mode_claim():
    result = online_metrics([0., .1, .2, .3, .4], [True, False, False, True, True], [True, False, False, False, True])
    assert result["missing_episode_counts"] == [0, 1, 0, 0]
    assert result["engineering_publication_recovery_lag_s"][0] == pytest.approx(.1)
    assert "unavailable" in result["true_motion_mode_metric"]
    with pytest.raises(ValueError, match="publication"):
        online_metrics([0., .1], [False, True], [True, True])
