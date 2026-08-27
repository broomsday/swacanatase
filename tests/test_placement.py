from pathlib import Path

import numpy as np

from swacanatase.placement import (
    DEFAULT_M_VALUES,
    central_inter_benzene_linker_anchor_pairs,
    central_para_linker_anchor_pairs,
    generate_m_equals_n_nanoring,
    generate_m_equals_n_nanorings,
    place_bp5_sidechains_around_nanoring,
)


def test_generate_requested_m_equals_n_nanorings() -> None:
    nanorings = generate_m_equals_n_nanorings(DEFAULT_M_VALUES)

    assert set(nanorings) == set(DEFAULT_M_VALUES)
    for m, nanoring in nanorings.items():
        assert nanoring.array_length() == 6 * m
        assert set(nanoring.res_name) == {"CNT"}
        assert set(nanoring.element) == {"C"}
        z_levels, band_counts = np.unique(
            np.round(nanoring.coord[:, 2], 6),
            return_counts=True,
        )
        assert z_levels.shape[0] == 3
        assert np.all(band_counts == 2 * m)


def test_central_inter_benzene_anchor_pairs_select_even_m_over_2_sites() -> None:
    for m in DEFAULT_M_VALUES:
        nanoring = generate_m_equals_n_nanoring(m)
        anchors = central_inter_benzene_linker_anchor_pairs(nanoring, count=m // 2)
        previous_phase_anchors = central_para_linker_anchor_pairs(
            nanoring,
            count=m // 2,
        )
        angles = np.array([anchor.angular_midpoint_degrees for anchor in anchors])
        previous_phase_angles = np.array(
            [anchor.angular_midpoint_degrees for anchor in previous_phase_anchors]
        )
        spacings = np.diff(np.r_[angles, angles[0] + 360.0])

        assert len(anchors) == m // 2
        assert np.allclose(spacings, 720.0 / m)
        assert np.allclose(angles - previous_phase_angles, 180.0 / m)
        assert all(
            anchor.atom_indices[0] == previous_anchor.atom_indices[1]
            for anchor, previous_anchor in zip(anchors, previous_phase_anchors, strict=True)
        )
        assert np.allclose(
            [anchor.ring_axis for anchor in anchors],
            np.array([0.0, 0.0, 1.0]),
        )
        assert np.allclose(
            [np.linalg.norm(anchor.radial_direction) for anchor in anchors],
            1.0,
        )
        assert np.allclose(
            [np.linalg.norm(anchor.tangential_direction) for anchor in anchors],
            1.0,
        )


def test_bp5_virtual_carbons_snap_to_selected_anchor_carbons() -> None:
    placement = place_bp5_sidechains_around_nanoring(
        m=18,
        cif_path=Path("data/rcsb/BP5.cif"),
    )

    assert placement.nanoring.array_length() == 108
    assert placement.sidechains.array_length() == 9 * 34
    assert placement.complex.array_length() == 108 + 9 * 34

    for residue_id, anchor in enumerate(placement.anchor_pairs, start=1):
        residue = placement.sidechains[placement.sidechains.res_id == residue_id]
        cv1 = residue.coord[residue.atom_name.tolist().index("CV1")]
        cv2 = residue.coord[residue.atom_name.tolist().index("CV2")]
        pd = residue.coord[residue.atom_name.tolist().index("PD")]

        assert np.allclose(cv1, anchor.coordinates[0])
        assert np.allclose(cv2, anchor.coordinates[1])
        assert np.dot(pd - anchor.midpoint, anchor.radial_direction) > 0


def test_bp5_complex_uses_bp5_chain_a_and_nanoring_chain_b() -> None:
    placement = place_bp5_sidechains_around_nanoring(
        m=18,
        cif_path=Path("data/rcsb/BP5.cif"),
    )

    bp5_atoms = placement.complex[placement.complex.res_name != "CNT"]
    nanoring_atoms = placement.complex[placement.complex.res_name == "CNT"]

    assert set(bp5_atoms.chain_id) == {"A"}
    assert set(nanoring_atoms.chain_id) == {"B"}
