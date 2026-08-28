from pathlib import Path

import numpy as np

from swacanatase.active_site import (
    DEFAULT_PD_C_BOND_LENGTH,
    build_bp5_palladium_active_site,
)
from swacanatase.placement import (
    DEFAULT_M_VALUES,
    central_inter_benzene_linker_anchor_pairs,
    central_para_linker_anchor_pairs,
    generate_m_equals_n_nanoring,
    generate_m_equals_n_nanorings,
    main as placement_main,
    place_bp5_sidechains_around_nanoring,
    write_bp5_nanoring_series,
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


def test_bp5_virtual_carbons_keep_ideal_geometry_after_best_anchor_fit() -> None:
    placement = place_bp5_sidechains_around_nanoring(
        m=18,
        cif_path=Path("data/rcsb/BP5.cif"),
    )
    source = build_bp5_palladium_active_site(Path("data/rcsb/BP5.cif"))
    source_cv_distance = np.linalg.norm(
        _atom_coord(source, "CV2") - _atom_coord(source, "CV1")
    )

    assert placement.nanoring.array_length() == 108
    assert placement.sidechains.array_length() == 9 * 34
    assert placement.complex.array_length() == 108 + 9 * 34

    for residue_id, anchor in enumerate(placement.anchor_pairs, start=1):
        residue = placement.sidechains[placement.sidechains.res_id == residue_id]
        cv1 = _atom_coord(residue, "CV1")
        cv2 = _atom_coord(residue, "CV2")
        pd = _atom_coord(residue, "PD")
        cv_midpoint = (cv1 + cv2) / 2.0
        expected_residual = (source_cv_distance - anchor.anchor_distance) / 2.0

        assert np.allclose(cv_midpoint, anchor.midpoint, atol=1e-6)
        assert np.allclose(np.linalg.norm(cv2 - cv1), source_cv_distance, atol=1e-6)
        assert np.allclose(
            _unit(cv2 - cv1),
            _unit(anchor.coordinates[1] - anchor.coordinates[0]),
            atol=1e-6,
        )
        assert np.isclose(
            np.linalg.norm(cv1 - anchor.coordinates[0]),
            expected_residual,
            atol=1e-6,
        )
        assert np.isclose(
            np.linalg.norm(cv2 - anchor.coordinates[1]),
            expected_residual,
            atol=1e-6,
        )
        assert np.isclose(np.linalg.norm(cv1 - pd), DEFAULT_PD_C_BOND_LENGTH, atol=1e-6)
        assert np.isclose(np.linalg.norm(cv2 - pd), DEFAULT_PD_C_BOND_LENGTH, atol=1e-6)
        assert np.dot(pd - anchor.midpoint, anchor.radial_direction) > 0


def test_bp5_virtual_carbons_can_still_snap_to_selected_anchor_carbons() -> None:
    placement = place_bp5_sidechains_around_nanoring(
        m=18,
        cif_path=Path("data/rcsb/BP5.cif"),
        snap_virtual_carbons=True,
    )

    for residue_id, anchor in enumerate(placement.anchor_pairs, start=1):
        residue = placement.sidechains[placement.sidechains.res_id == residue_id]

        assert np.allclose(_atom_coord(residue, "CV1"), anchor.coordinates[0])
        assert np.allclose(_atom_coord(residue, "CV2"), anchor.coordinates[1])


def test_bp5_complex_uses_bp5_chain_a_and_nanoring_chain_b() -> None:
    placement = place_bp5_sidechains_around_nanoring(
        m=18,
        cif_path=Path("data/rcsb/BP5.cif"),
    )

    bp5_atoms = placement.complex[placement.complex.res_name != "CNT"]
    nanoring_atoms = placement.complex[placement.complex.res_name == "CNT"]

    assert set(bp5_atoms.chain_id) == {"A"}
    assert set(nanoring_atoms.chain_id) == {"B"}


def test_write_bp5_nanoring_series_separates_nanoring_and_theozyme_outputs(
    tmp_path: Path,
) -> None:
    written_paths = write_bp5_nanoring_series(
        output_dir=tmp_path,
        m_values=(18,),
        overwrite=True,
    )

    assert written_paths == [
        tmp_path / "nanoring" / "nanoring_M18.cif",
        tmp_path / "theozyme" / "nanoring_M18_bp5.cif",
    ]
    assert all(path.exists() for path in written_paths)


def test_write_bp5_nanoring_series_can_write_rotamer_and_secondary_structure_outputs(
    tmp_path: Path,
) -> None:
    written_paths = write_bp5_nanoring_series(
        output_dir=tmp_path,
        m_values=(18,),
        overwrite=True,
        enumerate_bp5_rotamers=True,
        max_rotamers_per_site=1,
        secondary_structure="alpha_helix",
        residues_before=1,
        residues_after=1,
    )

    assert len(written_paths) == 2 + 1 + 1
    assert written_paths[:2] == [
        tmp_path / "nanoring" / "nanoring_M18.cif",
        tmp_path / "theozyme" / "nanoring_M18_bp5.cif",
    ]
    assert all(path.exists() for path in written_paths)
    assert sum(path.parent.name == "rotamers" for path in written_paths) == 1
    assert (
        sum(path.parent.name == "secondary_structure" for path in written_paths)
        == 1
    )
    assert tmp_path / "rotamers" / "nanoring_M18_gplus_m90.cif" in written_paths
    assert all(
        "_alpha_helix_pre1_post1" in path.stem
        for path in written_paths
        if path.parent.name == "secondary_structure"
    )


def test_placement_cli_exposes_rotamer_and_secondary_structure_output_modes(
    tmp_path: Path,
) -> None:
    exit_code = placement_main(
        [
            "--m",
            "18",
            "--output-dir",
            str(tmp_path),
            "--overwrite",
            "--enumerate-bp5-rotamers",
            "--max-rotamers-per-site",
            "1",
            "--secondary-structure",
            "beta_strand",
            "--residues-before",
            "1",
            "--residues-after",
            "0",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "rotamers").is_dir()
    assert (tmp_path / "secondary_structure").is_dir()
    assert len(list((tmp_path / "rotamers").glob("*.cif"))) == 1
    assert len(list((tmp_path / "secondary_structure").glob("*.cif"))) == 1


def _atom_coord(atom_array, atom_name: str) -> np.ndarray:
    return atom_array.coord[atom_array.atom_name.tolist().index(atom_name)]


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)
