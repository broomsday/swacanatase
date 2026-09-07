from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from itertools import product
from typing import Literal

import biotite.structure as struc
import numpy as np

from .bp5_rotamers import BP5RotamerPlacement
from .clashes import score_heavy_atom_clashes

SecondaryStructureType = Literal["alpha_helix", "beta_strand"]
TurnMotifType = Literal[
    "beta_turn_ad",
    "beta_turn_pd",
    "beta_turn_pa",
    "beta_turn_ad_prime",
    "beta_turn_pd_prime",
    "beta_turn_ab1",
    "beta_turn_ab2",
    "beta_turn_az",
    "beta_turn_ag",
    "beta_turn_bcis_p",
    "beta_turn_d_d",
    "beta_turn_pcis_d",
    "beta_turn_dn",
    "beta_turn_dd",
    "beta_turn_pcis_p",
    "beta_turn_cis_da",
    "beta_turn_pg",
    "beta_turn_cis_dp",
    "gamma_turn_inverse",
    "gamma_turn_classic",
]
RamachandranLevel = Literal["favored", "allowed"]
TurnMotifTorsionSource = Literal["mode", "medoid"]

BACKBONE_ATOM_NAMES = ("N", "CA", "C", "O")
BP5_TERMINAL_ATOM_NAMES = ("H", "H2", "OXT", "HXT")

N_CA_BOND_LENGTH = 1.458
CA_C_BOND_LENGTH = 1.525
C_N_BOND_LENGTH = 1.329
C_O_BOND_LENGTH = 1.229
C_OXT_BOND_LENGTH = 1.343
N_H_BOND_LENGTH = 1.010
OXT_HXT_BOND_LENGTH = 0.967

C_N_CA_ANGLE_DEGREES = 121.7
N_CA_C_ANGLE_DEGREES = 111.2
CA_C_N_ANGLE_DEGREES = 116.2
CA_C_O_ANGLE_DEGREES = 120.8
CA_C_OXT_ANGLE_DEGREES = 120.8
CA_N_H_ANGLE_DEGREES = 109.5
C_OXT_HXT_ANGLE_DEGREES = 109.5
TRANS_PEPTIDE_OMEGA_DEGREES = 180.0
CARBONYL_O_DIHEDRAL_DEGREES = 180.0


@dataclass(frozen=True)
class BackboneTorsionTargets:
    phi_degrees: float
    psi_degrees: float
    label: str = "ideal"
    ramachandran_level: str = "ideal"


@dataclass(frozen=True)
class ResidueTorsionTargets:
    residue_offset: int
    phi_degrees: float | None
    psi_degrees: float | None
    omega_after_degrees: float = TRANS_PEPTIDE_OMEGA_DEGREES


@dataclass(frozen=True)
class TurnMotifDefinition:
    motif_type: TurnMotifType
    length: int
    central_bp5_offsets: tuple[int, ...]
    residue_torsions: tuple[ResidueTorsionTargets, ...]
    label: str
    previous_name: str | None = None
    source: str = "classical"
    requires_cis_peptide: bool = False
    fixed_residue_names: tuple[tuple[int, str], ...] = ()
    torsion_source: TurnMotifTorsionSource = "mode"
    torsion_variant_label: str = "mode"
    mode_residue_torsions: tuple[ResidueTorsionTargets, ...] = ()
    medoid_residue_torsions: tuple[ResidueTorsionTargets, ...] = ()


@dataclass(frozen=True)
class RamachandranBasin:
    """Sparse basin approximation used for deterministic phi/psi scans."""

    secondary_structure_type: SecondaryStructureType
    name: str
    center_phi_degrees: float
    center_psi_degrees: float
    favored_phi_radius_degrees: float
    favored_psi_radius_degrees: float
    allowed_phi_radius_degrees: float
    allowed_psi_radius_degrees: float


@dataclass(frozen=True)
class SecondaryStructureSegment:
    secondary_structure_type: SecondaryStructureType
    atom_array: struc.AtomArray
    bp5_residue_id: int
    residues_before: int
    residues_after: int
    torsion_targets: BackboneTorsionTargets


@dataclass(frozen=True)
class TurnMotifSegment:
    motif_type: TurnMotifType
    atom_array: struc.AtomArray
    bp5_residue_id: int
    bp5_motif_offset: int
    motif_definition: TurnMotifDefinition


@dataclass(frozen=True)
class SecondaryStructureClashScore:
    scaffold_score: float
    bp5_score: float
    neighboring_backbone_score: float

    @property
    def total_overlap_score(self) -> float:
        return float(
            self.scaffold_score + self.bp5_score + self.neighboring_backbone_score
        )

    @property
    def passes(self) -> bool:
        return self.total_overlap_score == 0.0


@dataclass(frozen=True)
class NanoringCylinderIntrusionScore:
    """Atom-center intrusions into the finite inner cylinder of a nanoring."""

    radius: float
    z_min: float
    z_max: float
    intruding_atom_count: int
    total_intrusion_depth: float
    max_intrusion_depth: float

    @property
    def passes(self) -> bool:
        return self.intruding_atom_count == 0


@dataclass(frozen=True)
class SecondaryStructureOrientationMetrics:
    secondary_structure_direction: np.ndarray
    radial_alignment: float
    tangential_alignment: float
    axial_alignment: float
    n_terminal_exit_vector: np.ndarray
    c_terminal_exit_vector: np.ndarray


SECONDARY_STRUCTURE_TARGETS: dict[SecondaryStructureType, BackboneTorsionTargets] = {
    "alpha_helix": BackboneTorsionTargets(phi_degrees=-60.0, psi_degrees=-45.0),
    "beta_strand": BackboneTorsionTargets(phi_degrees=-135.0, psi_degrees=135.0),
}


def _beta_turn_residue_torsions(
    omega2_degrees: float,
    phi2_degrees: float,
    psi2_degrees: float,
    omega3_degrees: float,
    phi3_degrees: float,
    psi3_degrees: float,
    omega4_degrees: float,
) -> tuple[ResidueTorsionTargets, ...]:
    return (
        ResidueTorsionTargets(
            1,
            None,
            None,
            omega_after_degrees=omega2_degrees,
        ),
        ResidueTorsionTargets(
            2,
            phi2_degrees,
            psi2_degrees,
            omega_after_degrees=omega3_degrees,
        ),
        ResidueTorsionTargets(
            3,
            phi3_degrees,
            psi3_degrees,
            omega_after_degrees=omega4_degrees,
        ),
    )


def _beta_turn_definition(
    motif_type: TurnMotifType,
    label: str,
    previous_name: str,
    central_bp5_offsets: tuple[int, ...],
    mode_values: tuple[float, float, float, float, float, float, float],
    medoid_values: tuple[float, float, float, float, float, float, float],
    fixed_residue_names: tuple[tuple[int, str], ...] = (),
    requires_cis_peptide: bool = False,
) -> TurnMotifDefinition:
    mode_torsions = _beta_turn_residue_torsions(*mode_values)
    medoid_torsions = _beta_turn_residue_torsions(*medoid_values)
    return TurnMotifDefinition(
        motif_type=motif_type,
        length=4,
        central_bp5_offsets=central_bp5_offsets,
        residue_torsions=mode_torsions,
        label=label,
        previous_name=previous_name,
        source="BetaTurnLib18 v1.1",
        requires_cis_peptide=requires_cis_peptide,
        fixed_residue_names=fixed_residue_names,
        torsion_source="mode",
        torsion_variant_label="mode",
        mode_residue_torsions=mode_torsions,
        medoid_residue_torsions=medoid_torsions,
    )


DEFAULT_TURN_MOTIF_TYPES: tuple[TurnMotifType, ...] = (
    "beta_turn_ad",
    "beta_turn_ab1",
    "beta_turn_ab2",
    "gamma_turn_inverse",
)
TURN_MOTIF_DEFINITIONS: dict[TurnMotifType, TurnMotifDefinition] = {
    definition.motif_type: definition
    for definition in (
        _beta_turn_definition(
            "beta_turn_ad",
            "AD beta turn",
            "I",
            (2, 3),
            (185.12, -62.25, -23.48, 181.88, -96.25, -2.44, 179.02),
            (184.88, -65.27, -23.52, 182.83, -98.73, -18.06, 181.10),
        ),
        _beta_turn_definition(
            "beta_turn_pd",
            "Pd beta turn",
            "II",
            (2,),
            (179.57, -55.26, 133.10, 179.60, 91.18, -6.22, 180.91),
            (178.69, -55.28, 132.76, 175.53, 82.38, -2.88, 179.60),
        ),
        _beta_turn_definition(
            "beta_turn_pa",
            "Pa beta turn",
            "new_prev_II",
            (2,),
            (176.22, -59.93, 135.21, 178.77, 58.69, 27.54, 177.53),
            (180.97, -79.76, 149.11, 170.92, 61.47, 32.21, 180.14),
        ),
        _beta_turn_definition(
            "beta_turn_ad_prime",
            "ad beta turn",
            "I'",
            (2, 3),
            (182.81, 47.08, 44.65, 174.36, 82.66, 1.09, 182.92),
            (177.14, 54.08, 37.14, 174.19, 77.07, 7.66, 182.45),
        ),
        _beta_turn_definition(
            "beta_turn_ab1",
            "AB1 beta turn",
            "new_prev_VIII",
            (2, 3),
            (184.02, -67.20, -30.96, 172.54, -136.03, 162.29, 182.93),
            (178.29, -76.75, -33.07, 180.05, -137.67, 155.99, 179.53),
        ),
        _beta_turn_definition(
            "beta_turn_az",
            "AZ beta turn",
            "new_prev_VIII",
            (2, 3),
            (181.01, -74.11, -27.71, 182.08, -140.41, 75.09, 190.54),
            (181.51, -83.81, -17.76, 182.70, -128.88, 61.72, 177.81),
        ),
        _beta_turn_definition(
            "beta_turn_ab2",
            "AB2 beta turn",
            "VIII",
            (2, 3),
            (175.48, -69.29, -30.16, 169.41, -120.25, 128.00, 177.81),
            (179.88, -76.37, -33.97, 184.30, -116.48, 120.10, 183.38),
        ),
        _beta_turn_definition(
            "beta_turn_pd_prime",
            "pD beta turn",
            "II'",
            (3,),
            (179.73, 57.44, -130.18, 181.81, -95.43, 11.23, 181.96),
            (177.42, 56.16, -135.60, 183.17, -90.92, 4.81, 180.04),
        ),
        _beta_turn_definition(
            "beta_turn_ag",
            "AG beta turn",
            "new_prev_VIII",
            (2, 3),
            (179.98, -66.42, -19.26, 176.01, -82.48, 63.16, 183.48),
            (184.32, -71.78, -15.84, 187.19, -87.70, 74.65, 181.63),
        ),
        _beta_turn_definition(
            "beta_turn_bcis_p",
            "BcisP beta turn",
            "VIb",
            (2,),
            (178.76, -137.53, 119.38, 359.11, -66.49, 163.88, 179.76),
            (176.27, -127.05, 120.42, 358.00, -65.63, 161.27, 182.14),
            fixed_residue_names=((3, "PRO"),),
            requires_cis_peptide=True,
        ),
        _beta_turn_definition(
            "beta_turn_d_d",
            "dD beta turn",
            "new",
            (3,),
            (180.94, 94.08, -1.19, 186.08, -127.95, 15.44, 170.94),
            (182.57, 99.89, -17.34, 185.02, -113.78, 9.87, 180.65),
        ),
        _beta_turn_definition(
            "beta_turn_pcis_d",
            "PcisD beta turn",
            "VIa1",
            (2,),
            (175.04, -59.70, 144.37, 9.21, -92.77, 8.32, 175.56),
            (175.04, -59.70, 144.37, 9.21, -92.77, 8.32, 175.56),
            fixed_residue_names=((3, "PRO"),),
            requires_cis_peptide=True,
        ),
        _beta_turn_definition(
            "beta_turn_dn",
            "dN beta turn",
            "new",
            (3,),
            (183.90, 69.32, 8.93, 177.80, -131.73, -63.28, 185.11),
            (178.53, 76.06, -3.15, 179.31, -122.82, -50.46, 179.84),
        ),
        _beta_turn_definition(
            "beta_turn_dd",
            "Dd beta turn",
            "new",
            (2,),
            (182.75, -115.16, 15.69, 186.11, 100.65, -12.30, 176.58),
            (177.54, -99.08, 19.56, 173.80, 108.72, -14.53, 182.16),
        ),
        _beta_turn_definition(
            "beta_turn_pcis_p",
            "PcisP beta turn",
            "new_prev_VIb",
            (2,),
            (175.26, -66.01, 147.89, 0.25, -75.72, 142.11, 176.61),
            (183.55, -78.86, 145.45, 353.82, -78.06, 140.34, 178.16),
            fixed_residue_names=((3, "PRO"),),
            requires_cis_peptide=True,
        ),
        _beta_turn_definition(
            "beta_turn_cis_da",
            "cisDA beta turn",
            "new",
            (3,),
            (3.33, -94.18, 7.96, 187.90, -61.40, -38.41, 184.42),
            (0.14, -97.53, 3.93, 184.13, -66.86, -35.89, 182.02),
            fixed_residue_names=((2, "PRO"),),
            requires_cis_peptide=True,
        ),
        _beta_turn_definition(
            "beta_turn_pg",
            "pG beta turn",
            "new",
            (3,),
            (180.06, 73.58, -162.38, 180.27, -78.96, 77.17, 188.34),
            (188.70, 68.28, -139.60, 178.07, -79.92, 121.91, 175.54),
        ),
        _beta_turn_definition(
            "beta_turn_cis_dp",
            "cisDP beta turn",
            "new",
            (3,),
            (11.10, -86.47, 4.44, 169.99, -71.39, 157.80, 175.12),
            (11.10, -86.47, 4.44, 169.99, -71.39, 157.80, 175.12),
            fixed_residue_names=((2, "PRO"),),
            requires_cis_peptide=True,
        ),
        TurnMotifDefinition(
            motif_type="gamma_turn_inverse",
            length=3,
            central_bp5_offsets=(2,),
            residue_torsions=(ResidueTorsionTargets(2, -75.0, 65.0),),
            label="inverse gamma turn",
        ),
        TurnMotifDefinition(
            motif_type="gamma_turn_classic",
            length=3,
            central_bp5_offsets=(2,),
            residue_torsions=(ResidueTorsionTargets(2, 75.0, -65.0),),
            label="classic gamma turn",
        ),
    )
}

RAMACHANDRAN_DISALLOWED = 0
RAMACHANDRAN_ALLOWED = 1
RAMACHANDRAN_FAVORED = 2

# First-pass analytic scan windows around ideal alpha/beta centers. These are
# intentionally sparse and replaceable with residue-class density grids such as
# MolProbity/Top8000 ``rama8000-*.data`` files if those are hydrated locally
# later. Radii are kept conservative so favored scans avoid the low-density
# tails admitted by broad validation contours.
RAMACHANDRAN_BASINS: dict[SecondaryStructureType, RamachandranBasin] = {
    "alpha_helix": RamachandranBasin(
        secondary_structure_type="alpha_helix",
        name="alpha_r",
        center_phi_degrees=-60.0,
        center_psi_degrees=-45.0,
        favored_phi_radius_degrees=15.0,
        favored_psi_radius_degrees=15.0,
        allowed_phi_radius_degrees=25.0,
        allowed_psi_radius_degrees=30.0,
    ),
    "beta_strand": RamachandranBasin(
        secondary_structure_type="beta_strand",
        name="beta",
        center_phi_degrees=-135.0,
        center_psi_degrees=135.0,
        favored_phi_radius_degrees=30.0,
        favored_psi_radius_degrees=30.0,
        allowed_phi_radius_degrees=45.0,
        allowed_psi_radius_degrees=40.0,
    ),
}


def phi_psi_grid_values(step_degrees: float = 5.0) -> np.ndarray:
    """Return periodic Ramachandran grid coordinates from -180 to <180."""
    step = _validate_phi_psi_step(step_degrees)
    return np.arange(-180.0, 180.0, step, dtype=float)


def secondary_structure_phi_psi_scan_matrix(
    secondary_structure_type: SecondaryStructureType,
    step_degrees: float = 5.0,
) -> np.ndarray:
    """Return a sparse phi/psi mask for one ideal secondary-structure basin.

    Rows are phi values and columns are psi values, both ordered as
    ``phi_psi_grid_values(step_degrees)``. Values are 0 for disallowed,
    1 for allowed, and 2 for favored.
    """
    basin = _require_ramachandran_basin(secondary_structure_type)
    grid_values = phi_psi_grid_values(step_degrees)
    matrix = np.zeros((len(grid_values), len(grid_values)), dtype=np.uint8)
    for phi_index, phi_degrees in enumerate(grid_values):
        for psi_index, psi_degrees in enumerate(grid_values):
            matrix[phi_index, psi_index] = _classify_basin_phi_psi(
                basin=basin,
                phi_degrees=float(phi_degrees),
                psi_degrees=float(psi_degrees),
            )
    return matrix


def secondary_structure_phi_psi_scan_targets(
    secondary_structure_type: SecondaryStructureType,
    step_degrees: float = 5.0,
    ramachandran_level: RamachandranLevel = "favored",
) -> tuple[BackboneTorsionTargets, ...]:
    """Return deterministic sparse phi/psi targets around an ideal basin.

    Targets are sorted from the basin center outward so scan limits keep the
    most ideal-like torsions first.
    """
    if ramachandran_level not in {"favored", "allowed"}:
        raise ValueError("ramachandran_level must be 'favored' or 'allowed'")
    basin = _require_ramachandran_basin(secondary_structure_type)
    minimum_classification = (
        RAMACHANDRAN_FAVORED
        if ramachandran_level == "favored"
        else RAMACHANDRAN_ALLOWED
    )
    targets: list[BackboneTorsionTargets] = []
    for phi_degrees in phi_psi_grid_values(step_degrees):
        for psi_degrees in phi_psi_grid_values(step_degrees):
            classification = _classify_basin_phi_psi(
                basin=basin,
                phi_degrees=float(phi_degrees),
                psi_degrees=float(psi_degrees),
            )
            if classification < minimum_classification:
                continue
            level = (
                "favored"
                if classification == RAMACHANDRAN_FAVORED
                else "allowed"
            )
            targets.append(
                BackboneTorsionTargets(
                    phi_degrees=float(phi_degrees),
                    psi_degrees=float(psi_degrees),
                    label=_phi_psi_target_label(
                        phi_degrees=float(phi_degrees),
                        psi_degrees=float(psi_degrees),
                    ),
                    ramachandran_level=level,
                )
            )
    return tuple(
        sorted(
            targets,
            key=lambda target: (
                _elliptical_basin_distance(
                    basin=basin,
                    phi_degrees=target.phi_degrees,
                    psi_degrees=target.psi_degrees,
                    phi_radius_degrees=basin.favored_phi_radius_degrees,
                    psi_radius_degrees=basin.favored_psi_radius_degrees,
                ),
                abs(_signed_angle_delta(
                    target.phi_degrees,
                    basin.center_phi_degrees,
                )),
                abs(_signed_angle_delta(
                    target.psi_degrees,
                    basin.center_psi_degrees,
                )),
                target.phi_degrees,
                target.psi_degrees,
            ),
        )
    )


def build_regular_secondary_structure_segment(
    bp5_rotamer: BP5RotamerPlacement | struc.AtomArray,
    secondary_structure_type: SecondaryStructureType,
    residues_before: int,
    residues_after: int,
    chain_id: str = "A",
    starting_residue_id: int = 1,
    starting_atom_id: int = 1,
    torsion_targets: BackboneTorsionTargets | None = None,
) -> SecondaryStructureSegment:
    """Grow a deterministic regular backbone segment around a fixed BP5 frame."""
    if residues_before < 0 or residues_after < 0:
        raise ValueError("residue counts must be non-negative")
    if secondary_structure_type not in SECONDARY_STRUCTURE_TARGETS:
        raise ValueError(
            "secondary_structure_type must be 'alpha_helix' or 'beta_strand'"
        )

    bp5 = (
        bp5_rotamer.atom_array.copy()
        if isinstance(bp5_rotamer, BP5RotamerPlacement)
        else bp5_rotamer.copy()
    )
    _require_atom_names(bp5, ("N", "CA", "C", "O"))

    torsion_targets = (
        SECONDARY_STRUCTURE_TARGETS[secondary_structure_type]
        if torsion_targets is None
        else torsion_targets
    )
    bp5_residue_id = starting_residue_id + residues_before
    backbone_coords = _grow_backbone_coordinates(
        bp5=bp5,
        residues_before=residues_before,
        residues_after=residues_after,
        targets=torsion_targets,
    )

    arrays: list[struc.AtomArray] = []
    next_atom_id = starting_atom_id
    for residue_id in range(starting_residue_id, bp5_residue_id):
        local_residue_id = residue_id - starting_residue_id + 1
        previous_coords = backbone_coords.get(local_residue_id - 1)
        next_coords = backbone_coords.get(local_residue_id + 1)
        arrays.append(
            _new_backbone_residue(
                residue_id=residue_id,
                chain_id=chain_id,
                coordinates=backbone_coords[local_residue_id],
                previous_coordinates=previous_coords,
                next_coordinates=next_coords,
                starting_atom_id=next_atom_id,
            )
        )
        next_atom_id += arrays[-1].array_length()

    prepared_bp5 = _prepare_bp5_residue(
        bp5=bp5,
        residue_id=bp5_residue_id,
        chain_id=chain_id,
        starting_atom_id=next_atom_id,
        has_previous=residues_before > 0,
        has_next=residues_after > 0,
        previous_coordinates=backbone_coords.get(residues_before),
    )
    arrays.append(prepared_bp5)
    next_atom_id += prepared_bp5.array_length()

    for residue_id in range(bp5_residue_id + 1, bp5_residue_id + residues_after + 1):
        local_residue_id = residue_id - starting_residue_id + 1
        previous_coords = backbone_coords.get(local_residue_id - 1)
        next_coords = backbone_coords.get(local_residue_id + 1)
        arrays.append(
            _new_backbone_residue(
                residue_id=residue_id,
                chain_id=chain_id,
                coordinates=backbone_coords[local_residue_id],
                previous_coordinates=previous_coords,
                next_coordinates=next_coords,
                starting_atom_id=next_atom_id,
            )
        )
        next_atom_id += arrays[-1].array_length()

    return SecondaryStructureSegment(
        secondary_structure_type=secondary_structure_type,
        atom_array=struc.concatenate(arrays),
        bp5_residue_id=bp5_residue_id,
        residues_before=residues_before,
        residues_after=residues_after,
        torsion_targets=torsion_targets,
    )


def build_turn_motif_segment(
    bp5_rotamer: BP5RotamerPlacement | struc.AtomArray,
    motif_type: TurnMotifType,
    bp5_motif_offset: int,
    chain_id: str = "A",
    starting_residue_id: int = 1,
    starting_atom_id: int = 1,
    motif_definition: TurnMotifDefinition | None = None,
) -> TurnMotifSegment:
    """Grow a finite beta/gamma-turn motif around a fixed BP5 residue."""
    motif_definition = (
        require_turn_motif_definition(motif_type)
        if motif_definition is None
        else motif_definition
    )
    if motif_definition.motif_type != motif_type:
        raise ValueError("motif_definition.motif_type must match motif_type")
    if bp5_motif_offset not in motif_definition.central_bp5_offsets:
        raise ValueError(
            f"bp5_motif_offset must be one of "
            f"{motif_definition.central_bp5_offsets} for {motif_type}"
        )
    fixed_residue_names = _turn_motif_fixed_residue_names(motif_definition)
    if bp5_motif_offset in fixed_residue_names:
        raise ValueError(
            f"BP5 cannot replace fixed {fixed_residue_names[bp5_motif_offset]} "
            f"residue {bp5_motif_offset} in {motif_type}"
        )

    bp5 = (
        bp5_rotamer.atom_array.copy()
        if isinstance(bp5_rotamer, BP5RotamerPlacement)
        else bp5_rotamer.copy()
    )
    _require_atom_names(bp5, ("N", "CA", "C", "O"))

    bp5_residue_id = starting_residue_id + bp5_motif_offset - 1
    torsions_by_offset = _turn_motif_torsions_by_offset(motif_definition)
    backbone_coords = _grow_backbone_coordinates_from_residue_torsions(
        bp5=bp5,
        bp5_local_residue_id=bp5_motif_offset,
        total_residues=motif_definition.length,
        torsions_by_residue_id=torsions_by_offset,
    )

    arrays: list[struc.AtomArray] = []
    next_atom_id = starting_atom_id
    for local_residue_id in range(1, motif_definition.length + 1):
        residue_id = starting_residue_id + local_residue_id - 1
        previous_coords = backbone_coords.get(local_residue_id - 1)
        next_coords = backbone_coords.get(local_residue_id + 1)
        if local_residue_id == bp5_motif_offset:
            prepared_bp5 = _prepare_bp5_residue(
                bp5=bp5,
                residue_id=residue_id,
                chain_id=chain_id,
                starting_atom_id=next_atom_id,
                has_previous=local_residue_id > 1,
                has_next=local_residue_id < motif_definition.length,
                previous_coordinates=previous_coords,
            )
            arrays.append(prepared_bp5)
        else:
            arrays.append(
                _new_backbone_residue(
                    residue_id=residue_id,
                    chain_id=chain_id,
                    coordinates=backbone_coords[local_residue_id],
                    previous_coordinates=previous_coords,
                    next_coordinates=next_coords,
                    starting_atom_id=next_atom_id,
                    res_name=fixed_residue_names.get(local_residue_id, "GLY"),
                )
            )
        next_atom_id += arrays[-1].array_length()

    return TurnMotifSegment(
        motif_type=motif_type,
        atom_array=struc.concatenate(arrays),
        bp5_residue_id=bp5_residue_id,
        bp5_motif_offset=bp5_motif_offset,
        motif_definition=motif_definition,
    )


def require_turn_motif_definition(
    motif_type: TurnMotifType,
) -> TurnMotifDefinition:
    try:
        return TURN_MOTIF_DEFINITIONS[motif_type]
    except KeyError as error:
        raise ValueError(f"unknown turn motif type {motif_type!r}") from error


def turn_motif_definition_with_torsion_source(
    motif_type: TurnMotifType,
    torsion_source: TurnMotifTorsionSource = "mode",
) -> TurnMotifDefinition:
    """Return a motif definition using either modal or medoid torsions."""
    if torsion_source not in {"mode", "medoid"}:
        raise ValueError("torsion_source must be 'mode' or 'medoid'")
    motif_definition = require_turn_motif_definition(motif_type)
    if torsion_source == "medoid" and motif_definition.medoid_residue_torsions:
        residue_torsions = motif_definition.medoid_residue_torsions
        selected_source: TurnMotifTorsionSource = "medoid"
    elif motif_definition.mode_residue_torsions:
        residue_torsions = motif_definition.mode_residue_torsions
        selected_source = "mode"
    else:
        residue_torsions = motif_definition.residue_torsions
        selected_source = "mode"
    return replace(
        motif_definition,
        residue_torsions=residue_torsions,
        torsion_source=selected_source,
        torsion_variant_label=selected_source,
    )


def turn_motif_perturbation_definitions(
    motif_definition: TurnMotifDefinition,
    step_degrees: float | None = None,
    radius_degrees: float = 0.0,
) -> tuple[TurnMotifDefinition, ...]:
    """Return deterministic phi/psi perturbations around a selected motif."""
    if radius_degrees < 0.0:
        raise ValueError("radius_degrees must be non-negative")
    if step_degrees is None or np.isclose(radius_degrees, 0.0):
        return (motif_definition,)
    if step_degrees <= 0.0:
        raise ValueError("step_degrees must be positive")

    offsets = np.arange(
        -radius_degrees,
        radius_degrees + step_degrees / 2.0,
        step_degrees,
    )
    perturbable_targets = tuple(
        targets
        for targets in motif_definition.residue_torsions
        if targets.phi_degrees is not None and targets.psi_degrees is not None
    )
    delta_sets = tuple(
        delta_set
        for delta_set in product(offsets.tolist(), repeat=2 * len(perturbable_targets))
        if _turn_torsion_delta_within_radius(delta_set, radius_degrees)
    )
    ordered_delta_sets = sorted(
        delta_sets,
        key=lambda delta_set: (
            sum(abs(delta) for delta in delta_set),
            sum(delta * delta for delta in delta_set),
            delta_set,
        ),
    )
    return tuple(
        _perturbed_turn_motif_definition(
            motif_definition=motif_definition,
            delta_set=delta_set,
        )
        for delta_set in ordered_delta_sets
    )


def score_secondary_structure_segment_clashes(
    segment: SecondaryStructureSegment | TurnMotifSegment,
    nanoring: struc.AtomArray | None = None,
    bp5_context: struc.AtomArray | None = None,
    neighboring_segments: Iterable[SecondaryStructureSegment | TurnMotifSegment] = (),
) -> SecondaryStructureClashScore:
    """Score post-growth backbone clashes against scaffold, BP5, and neighbors."""
    backbone = _backbone_atoms(segment.atom_array)
    generated_backbone = _generated_backbone_atoms(segment)
    scaffold_score = (
        score_heavy_atom_clashes(
            backbone,
            other=nanoring,
            ignore_same_residue=True,
            ignore_inter_residue_backbone_n_c=True,
        ).total_overlap_score
        if nanoring is not None
        else 0.0
    )
    bp5_score = (
        score_heavy_atom_clashes(
            generated_backbone,
            other=bp5_context,
            ignore_same_residue=True,
            ignore_inter_residue_backbone_n_c=True,
        ).total_overlap_score
        if bp5_context is not None
        else 0.0
    )
    neighboring_backbone_score = float(
        sum(
            score_heavy_atom_clashes(
                backbone,
                other=_backbone_atoms(neighboring_segment.atom_array),
                ignore_same_residue=True,
                ignore_inter_residue_backbone_n_c=True,
            ).total_overlap_score
            for neighboring_segment in neighboring_segments
        )
    )
    return SecondaryStructureClashScore(
        scaffold_score=float(scaffold_score),
        bp5_score=float(bp5_score),
        neighboring_backbone_score=neighboring_backbone_score,
    )


def score_nanoring_cylinder_intrusions(
    atom_array: struc.AtomArray,
    nanoring: struc.AtomArray,
    cylinder_radius: float | None = None,
    tolerance: float = 1e-6,
) -> NanoringCylinderIntrusionScore:
    """Count atom centers inside the finite cylinder wrapped by a nanoring.

    The current nanoring workflow keeps scaffolds aligned to global Z, so the
    inferred cylinder uses the nanoring z extent and the innermost carbon radial
    distance from the global Z axis.
    """
    if nanoring.array_length() == 0:
        raise ValueError("nanoring must contain atoms")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    radius = (
        _infer_nanoring_cylinder_radius(nanoring)
        if cylinder_radius is None
        else float(cylinder_radius)
    )
    if radius <= 0.0:
        raise ValueError("cylinder_radius must be positive")

    z_min = float(np.min(nanoring.coord[:, 2]))
    z_max = float(np.max(nanoring.coord[:, 2]))
    if z_max <= z_min:
        raise ValueError("nanoring must span a non-zero z extent")

    if atom_array.array_length() == 0:
        return NanoringCylinderIntrusionScore(
            radius=radius,
            z_min=z_min,
            z_max=z_max,
            intruding_atom_count=0,
            total_intrusion_depth=0.0,
            max_intrusion_depth=0.0,
        )

    radial_distances = np.linalg.norm(atom_array.coord[:, :2], axis=1)
    z_values = atom_array.coord[:, 2]
    intrudes = (
        (radial_distances < radius - tolerance)
        & (z_values > z_min + tolerance)
        & (z_values < z_max - tolerance)
    )
    intrusion_depths = radius - radial_distances[intrudes]
    return NanoringCylinderIntrusionScore(
        radius=radius,
        z_min=z_min,
        z_max=z_max,
        intruding_atom_count=int(intrudes.sum()),
        total_intrusion_depth=float(intrusion_depths.sum()),
        max_intrusion_depth=(
            float(intrusion_depths.max()) if intrusion_depths.size else 0.0
        ),
    )


def measure_secondary_structure_orientation(
    segment: SecondaryStructureSegment | TurnMotifSegment,
    radial_direction: np.ndarray,
    tangential_direction: np.ndarray,
    ring_axis: np.ndarray,
) -> SecondaryStructureOrientationMetrics:
    """Measure segment and terminal-exit directions in a nanoring anchor frame."""
    secondary_structure_direction = _secondary_structure_direction(segment)
    radial_axis = _unit(np.asarray(radial_direction, dtype=float))
    tangential_axis = _unit(np.asarray(tangential_direction, dtype=float))
    axial_axis = _unit(np.asarray(ring_axis, dtype=float))
    n_terminal_exit_vector, c_terminal_exit_vector = _terminal_exit_vectors(segment)
    return SecondaryStructureOrientationMetrics(
        secondary_structure_direction=secondary_structure_direction,
        radial_alignment=float(np.dot(secondary_structure_direction, radial_axis)),
        tangential_alignment=float(
            np.dot(secondary_structure_direction, tangential_axis)
        ),
        axial_alignment=float(np.dot(secondary_structure_direction, axial_axis)),
        n_terminal_exit_vector=n_terminal_exit_vector,
        c_terminal_exit_vector=c_terminal_exit_vector,
    )


def _require_ramachandran_basin(
    secondary_structure_type: SecondaryStructureType,
) -> RamachandranBasin:
    try:
        return RAMACHANDRAN_BASINS[secondary_structure_type]
    except KeyError as error:
        raise ValueError(
            "secondary_structure_type must be 'alpha_helix' or 'beta_strand'"
        ) from error


def _validate_phi_psi_step(step_degrees: float) -> float:
    step = float(step_degrees)
    if step <= 0.0:
        raise ValueError("step_degrees must be positive")
    step_count = 360.0 / step
    if not np.isclose(step_count, round(step_count)):
        raise ValueError("step_degrees must evenly divide 360 degrees")
    return step


def _classify_basin_phi_psi(
    basin: RamachandranBasin,
    phi_degrees: float,
    psi_degrees: float,
) -> int:
    if (
        _elliptical_basin_distance(
            basin=basin,
            phi_degrees=phi_degrees,
            psi_degrees=psi_degrees,
            phi_radius_degrees=basin.favored_phi_radius_degrees,
            psi_radius_degrees=basin.favored_psi_radius_degrees,
        )
        <= 1.0
    ):
        return RAMACHANDRAN_FAVORED
    if (
        _elliptical_basin_distance(
            basin=basin,
            phi_degrees=phi_degrees,
            psi_degrees=psi_degrees,
            phi_radius_degrees=basin.allowed_phi_radius_degrees,
            psi_radius_degrees=basin.allowed_psi_radius_degrees,
        )
        <= 1.0
    ):
        return RAMACHANDRAN_ALLOWED
    return RAMACHANDRAN_DISALLOWED


def _elliptical_basin_distance(
    basin: RamachandranBasin,
    phi_degrees: float,
    psi_degrees: float,
    phi_radius_degrees: float,
    psi_radius_degrees: float,
) -> float:
    phi_delta = _signed_angle_delta(phi_degrees, basin.center_phi_degrees)
    psi_delta = _signed_angle_delta(psi_degrees, basin.center_psi_degrees)
    return float(
        np.sqrt(
            (phi_delta / phi_radius_degrees) ** 2
            + (psi_delta / psi_radius_degrees) ** 2
        )
    )


def _phi_psi_target_label(phi_degrees: float, psi_degrees: float) -> str:
    return (
        f"phi_{_signed_angle_label_component(phi_degrees)}_"
        f"psi_{_signed_angle_label_component(psi_degrees)}"
    )


def _signed_angle_label_component(angle_degrees: float) -> str:
    rounded = int(round(_normalize_signed_angle(angle_degrees)))
    prefix = "p" if rounded >= 0 else "m"
    return f"{prefix}{abs(rounded):03d}"


def _normalize_signed_angle(angle_degrees: float) -> float:
    normalized = (angle_degrees + 180.0) % 360.0 - 180.0
    return 180.0 if np.isclose(normalized, -180.0) else float(normalized)


def _signed_angle_delta(angle_degrees: float, reference_degrees: float) -> float:
    return float((angle_degrees - reference_degrees + 180.0) % 360.0 - 180.0)


def _grow_backbone_coordinates(
    bp5: struc.AtomArray,
    residues_before: int,
    residues_after: int,
    targets: BackboneTorsionTargets,
) -> dict[int, dict[str, np.ndarray]]:
    bp5_residue_id = residues_before + 1
    total_residues = residues_before + 1 + residues_after
    torsions_by_residue_id = {
        residue_id: ResidueTorsionTargets(
            residue_offset=residue_id,
            phi_degrees=targets.phi_degrees,
            psi_degrees=targets.psi_degrees,
        )
        for residue_id in range(1, total_residues + 1)
    }
    return _grow_backbone_coordinates_from_residue_torsions(
        bp5=bp5,
        bp5_local_residue_id=bp5_residue_id,
        total_residues=total_residues,
        torsions_by_residue_id=torsions_by_residue_id,
    )


def _turn_motif_torsions_by_offset(
    motif_definition: TurnMotifDefinition,
) -> dict[int, ResidueTorsionTargets]:
    defined_targets = {
        target.residue_offset: target
        for target in motif_definition.residue_torsions
    }
    fallback = SECONDARY_STRUCTURE_TARGETS["beta_strand"]
    return {
        residue_offset: ResidueTorsionTargets(
            residue_offset=residue_offset,
            phi_degrees=(
                defined_targets[residue_offset].phi_degrees
                if (
                    residue_offset in defined_targets
                    and defined_targets[residue_offset].phi_degrees is not None
                )
                else fallback.phi_degrees
            ),
            psi_degrees=(
                defined_targets[residue_offset].psi_degrees
                if (
                    residue_offset in defined_targets
                    and defined_targets[residue_offset].psi_degrees is not None
                )
                else fallback.psi_degrees
            ),
            omega_after_degrees=(
                defined_targets[residue_offset].omega_after_degrees
                if residue_offset in defined_targets
                else TRANS_PEPTIDE_OMEGA_DEGREES
            ),
        )
        for residue_offset in range(1, motif_definition.length + 1)
    }


def _turn_motif_fixed_residue_names(
    motif_definition: TurnMotifDefinition,
) -> dict[int, str]:
    fixed_residue_names = {
        residue_offset: residue_name
        for residue_offset, residue_name in motif_definition.fixed_residue_names
    }
    invalid_offsets = [
        residue_offset
        for residue_offset in fixed_residue_names
        if residue_offset < 1 or residue_offset > motif_definition.length
    ]
    if invalid_offsets:
        raise ValueError(
            f"fixed residue offsets must be within 1..{motif_definition.length}: "
            f"{invalid_offsets}"
        )
    return fixed_residue_names


def _turn_torsion_delta_within_radius(
    delta_set: tuple[float, ...],
    radius_degrees: float,
) -> bool:
    return all(abs(delta) <= radius_degrees for delta in delta_set)


def _perturbed_turn_motif_definition(
    motif_definition: TurnMotifDefinition,
    delta_set: tuple[float, ...],
) -> TurnMotifDefinition:
    deltas = iter(delta_set)
    perturbed_torsions: list[ResidueTorsionTargets] = []
    label_parts: list[str] = [motif_definition.torsion_source]
    for targets in motif_definition.residue_torsions:
        if targets.phi_degrees is None or targets.psi_degrees is None:
            perturbed_torsions.append(targets)
            continue
        phi_delta = float(next(deltas))
        psi_delta = float(next(deltas))
        perturbed_torsions.append(
            replace(
                targets,
                phi_degrees=_normalize_signed_angle(
                    targets.phi_degrees + phi_delta,
                ),
                psi_degrees=_normalize_signed_angle(
                    targets.psi_degrees + psi_delta,
                ),
            )
        )
        label_parts.extend(
            (
                f"r{targets.residue_offset}",
                f"dphi{_signed_angle_label_component(phi_delta)}",
                f"dpsi{_signed_angle_label_component(psi_delta)}",
            )
        )
    variant_label = (
        motif_definition.torsion_source
        if all(np.isclose(delta, 0.0) for delta in delta_set)
        else "_".join(label_parts)
    )
    return replace(
        motif_definition,
        residue_torsions=tuple(perturbed_torsions),
        torsion_variant_label=variant_label,
    )


def _grow_backbone_coordinates_from_residue_torsions(
    bp5: struc.AtomArray,
    bp5_local_residue_id: int,
    total_residues: int,
    torsions_by_residue_id: dict[int, ResidueTorsionTargets],
) -> dict[int, dict[str, np.ndarray]]:
    bp5_coords = {
        atom_name: _atom_coord(bp5, atom_name).copy()
        for atom_name in BACKBONE_ATOM_NAMES
    }
    coordinates: dict[int, dict[str, np.ndarray]] = {
        bp5_local_residue_id: bp5_coords
    }

    for residue_id in range(bp5_local_residue_id - 1, 0, -1):
        next_coords = coordinates[residue_id + 1]
        residue_targets = torsions_by_residue_id[residue_id]
        next_residue_targets = torsions_by_residue_id[residue_id + 1]
        c_coord = _place_internal_coordinate_atom(
            atom_1=next_coords["C"],
            atom_2=next_coords["CA"],
            atom_3=next_coords["N"],
            bond_length=C_N_BOND_LENGTH,
            bond_angle_degrees=C_N_CA_ANGLE_DEGREES,
            dihedral_degrees=_required_phi(next_residue_targets),
        )
        ca_coord = _place_internal_coordinate_atom(
            atom_1=next_coords["CA"],
            atom_2=next_coords["N"],
            atom_3=c_coord,
            bond_length=CA_C_BOND_LENGTH,
            bond_angle_degrees=CA_C_N_ANGLE_DEGREES,
            dihedral_degrees=residue_targets.omega_after_degrees,
        )
        n_coord = _place_internal_coordinate_atom(
            atom_1=next_coords["N"],
            atom_2=c_coord,
            atom_3=ca_coord,
            bond_length=N_CA_BOND_LENGTH,
            bond_angle_degrees=N_CA_C_ANGLE_DEGREES,
            dihedral_degrees=_required_psi(residue_targets),
        )
        o_coord = _place_carbonyl_oxygen(n_coord, ca_coord, c_coord)
        coordinates[residue_id] = {
            "N": n_coord,
            "CA": ca_coord,
            "C": c_coord,
            "O": o_coord,
        }

    for residue_id in range(bp5_local_residue_id + 1, total_residues + 1):
        previous_coords = coordinates[residue_id - 1]
        previous_residue_targets = torsions_by_residue_id[residue_id - 1]
        residue_targets = torsions_by_residue_id[residue_id]
        n_coord = _place_internal_coordinate_atom(
            atom_1=previous_coords["N"],
            atom_2=previous_coords["CA"],
            atom_3=previous_coords["C"],
            bond_length=C_N_BOND_LENGTH,
            bond_angle_degrees=CA_C_N_ANGLE_DEGREES,
            dihedral_degrees=_required_psi(previous_residue_targets),
        )
        ca_coord = _place_internal_coordinate_atom(
            atom_1=previous_coords["CA"],
            atom_2=previous_coords["C"],
            atom_3=n_coord,
            bond_length=N_CA_BOND_LENGTH,
            bond_angle_degrees=C_N_CA_ANGLE_DEGREES,
            dihedral_degrees=previous_residue_targets.omega_after_degrees,
        )
        c_coord = _place_internal_coordinate_atom(
            atom_1=previous_coords["C"],
            atom_2=n_coord,
            atom_3=ca_coord,
            bond_length=CA_C_BOND_LENGTH,
            bond_angle_degrees=N_CA_C_ANGLE_DEGREES,
            dihedral_degrees=_required_phi(residue_targets),
        )
        o_coord = _place_carbonyl_oxygen(n_coord, ca_coord, c_coord)
        coordinates[residue_id] = {
            "N": n_coord,
            "CA": ca_coord,
            "C": c_coord,
            "O": o_coord,
        }

    return coordinates


def _required_phi(targets: ResidueTorsionTargets) -> float:
    if targets.phi_degrees is None:
        raise ValueError(f"residue {targets.residue_offset} is missing phi target")
    return targets.phi_degrees


def _required_psi(targets: ResidueTorsionTargets) -> float:
    if targets.psi_degrees is None:
        raise ValueError(f"residue {targets.residue_offset} is missing psi target")
    return targets.psi_degrees


def _place_carbonyl_oxygen(
    n_coord: np.ndarray,
    ca_coord: np.ndarray,
    c_coord: np.ndarray,
) -> np.ndarray:
    return _place_internal_coordinate_atom(
        atom_1=n_coord,
        atom_2=ca_coord,
        atom_3=c_coord,
        bond_length=C_O_BOND_LENGTH,
        bond_angle_degrees=CA_C_O_ANGLE_DEGREES,
        dihedral_degrees=CARBONYL_O_DIHEDRAL_DEGREES,
    )


def _place_internal_coordinate_atom(
    atom_1: np.ndarray,
    atom_2: np.ndarray,
    atom_3: np.ndarray,
    bond_length: float,
    bond_angle_degrees: float,
    dihedral_degrees: float,
) -> np.ndarray:
    axis = _unit(atom_3 - atom_2)
    normal = _unit(np.cross(atom_2 - atom_1, axis))
    perpendicular = np.cross(normal, axis)
    bond_angle = np.radians(bond_angle_degrees)
    dihedral = np.radians(dihedral_degrees)
    direction = (
        np.cos(np.pi - bond_angle) * axis
        + np.sin(np.pi - bond_angle)
        * (np.cos(dihedral) * perpendicular + np.sin(dihedral) * normal)
    )
    return atom_3 + bond_length * direction


def _new_backbone_residue(
    residue_id: int,
    chain_id: str,
    coordinates: dict[str, np.ndarray],
    previous_coordinates: dict[str, np.ndarray] | None,
    next_coordinates: dict[str, np.ndarray] | None,
    starting_atom_id: int,
    res_name: str = "GLY",
) -> struc.AtomArray:
    atom_names = list(BACKBONE_ATOM_NAMES)
    elements = ["N", "C", "C", "O"]
    coords = [coordinates[name] for name in BACKBONE_ATOM_NAMES]

    for atom_name, element, coord in _peptide_cap_atom_records(
        coordinates=coordinates,
        previous_coordinates=previous_coordinates,
        next_coordinates=next_coordinates,
        include_internal_n_hydrogen=res_name != "PRO",
    ):
        atom_names.append(atom_name)
        elements.append(element)
        coords.append(coord)

    atom_count = len(atom_names)
    atoms = struc.AtomArray(atom_count)
    atoms.coord = np.array(coords)
    atoms.chain_id = np.full(atom_count, chain_id, dtype="U4")
    atoms.res_id = np.full(atom_count, residue_id, dtype=int)
    atoms.ins_code = np.full(atom_count, "", dtype="U1")
    atoms.res_name = np.full(atom_count, res_name, dtype="U5")
    atoms.hetero = np.zeros(atom_count, dtype=bool)
    atoms.atom_name = np.array(atom_names, dtype="U6")
    atoms.element = np.array(elements, dtype="U2")
    atoms.set_annotation(
        "atom_id",
        np.arange(starting_atom_id, starting_atom_id + atom_count, dtype=int),
    )
    atoms.set_annotation("occupancy", np.ones(atom_count, dtype=float))
    atoms.set_annotation("b_factor", np.zeros(atom_count, dtype=float))
    return atoms


def _prepare_bp5_residue(
    bp5: struc.AtomArray,
    residue_id: int,
    chain_id: str,
    starting_atom_id: int,
    has_previous: bool,
    has_next: bool,
    previous_coordinates: dict[str, np.ndarray] | None,
) -> struc.AtomArray:
    if has_previous and previous_coordinates is None:
        raise ValueError("previous_coordinates are required for internal BP5 N-H")

    prepared = bp5[~np.isin(bp5.atom_name, list(BP5_TERMINAL_ATOM_NAMES))].copy()
    terminal_records = _peptide_cap_atom_records(
        coordinates={
            atom_name: _atom_coord(bp5, atom_name).copy()
            for atom_name in BACKBONE_ATOM_NAMES
        },
        previous_coordinates=previous_coordinates,
        next_coordinates={} if has_next else None,
    )
    if terminal_records:
        prepared = struc.concatenate(
            [
                prepared,
                _new_terminal_atom_array(
                    atom_records=terminal_records,
                    residue_id=int(prepared.res_id[0]),
                    chain_id=str(prepared.chain_id[0]),
                    res_name=str(prepared.res_name[0]),
                    hetero=bool(prepared.hetero[0]),
                    starting_atom_id=1,
                ),
            ]
        )
    prepared.chain_id = np.full(prepared.array_length(), chain_id, dtype="U4")
    prepared.res_id = np.full(prepared.array_length(), residue_id, dtype=int)
    prepared.set_annotation(
        "atom_id",
        np.arange(
            starting_atom_id,
            starting_atom_id + prepared.array_length(),
            dtype=int,
        ),
    )
    return prepared


def _peptide_cap_atom_records(
    coordinates: dict[str, np.ndarray],
    previous_coordinates: dict[str, np.ndarray] | None,
    next_coordinates: dict[str, np.ndarray] | None,
    include_internal_n_hydrogen: bool = True,
) -> list[tuple[str, str, np.ndarray]]:
    atom_records: list[tuple[str, str, np.ndarray]] = []
    n_coord = coordinates["N"]
    ca_coord = coordinates["CA"]
    c_coord = coordinates["C"]
    o_coord = coordinates["O"]

    if previous_coordinates is None:
        h_coord, h2_coord = _terminal_n_hydrogen_coords(
            n_coord=n_coord,
            ca_coord=ca_coord,
            c_coord=c_coord,
        )
        atom_records.extend(
            [
                ("H", "H", h_coord),
                ("H2", "H", h2_coord),
            ]
        )
    elif include_internal_n_hydrogen:
        atom_records.append(
            (
                "H",
                "H",
                _internal_n_hydrogen_coord(
                    previous_c_coord=previous_coordinates["C"],
                    n_coord=n_coord,
                    ca_coord=ca_coord,
                ),
            )
        )

    if next_coordinates is None:
        oxt_coord = _terminal_oxt_coord(
            ca_coord=ca_coord,
            c_coord=c_coord,
            o_coord=o_coord,
        )
        hxt_coord = _terminal_hxt_coord(
            ca_coord=ca_coord,
            c_coord=c_coord,
            o_coord=o_coord,
            oxt_coord=oxt_coord,
        )
        atom_records.extend(
            [
                ("OXT", "O", oxt_coord),
                ("HXT", "H", hxt_coord),
            ]
        )

    return atom_records


def _new_terminal_atom_array(
    atom_records: list[tuple[str, str, np.ndarray]],
    residue_id: int,
    chain_id: str,
    res_name: str,
    hetero: bool,
    starting_atom_id: int,
) -> struc.AtomArray:
    atoms = struc.AtomArray(len(atom_records))
    atoms.coord = np.array([coord for _, _, coord in atom_records], dtype=float)
    atoms.chain_id = np.full(len(atom_records), chain_id, dtype="U4")
    atoms.res_id = np.full(len(atom_records), residue_id, dtype=int)
    atoms.ins_code = np.full(len(atom_records), "", dtype="U1")
    atoms.res_name = np.full(len(atom_records), res_name, dtype="U5")
    atoms.hetero = np.full(len(atom_records), hetero, dtype=bool)
    atoms.atom_name = np.array([name for name, _, _ in atom_records], dtype="U6")
    atoms.element = np.array([element for _, element, _ in atom_records], dtype="U2")
    atoms.set_annotation(
        "atom_id",
        np.arange(starting_atom_id, starting_atom_id + len(atom_records), dtype=int),
    )
    atoms.set_annotation("occupancy", np.ones(len(atom_records), dtype=float))
    atoms.set_annotation("b_factor", np.zeros(len(atom_records), dtype=float))
    return atoms


def _internal_n_hydrogen_coord(
    previous_c_coord: np.ndarray,
    n_coord: np.ndarray,
    ca_coord: np.ndarray,
) -> np.ndarray:
    direction = -_unit(
        _unit(previous_c_coord - n_coord)
        + _unit(ca_coord - n_coord)
    )
    return n_coord + N_H_BOND_LENGTH * direction


def _terminal_n_hydrogen_coords(
    n_coord: np.ndarray,
    ca_coord: np.ndarray,
    c_coord: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ca_axis = _unit(ca_coord - n_coord)
    reference = c_coord - ca_coord
    perpendicular = _unit(reference - np.dot(reference, ca_axis) * ca_axis)
    orthogonal = np.cross(ca_axis, perpendicular)
    theta = np.radians(CA_N_H_ANGLE_DEGREES)
    first_direction = (
        np.cos(theta) * ca_axis
        + np.sin(theta)
        * (
            np.cos(np.radians(120.0)) * perpendicular
            + np.sin(np.radians(120.0)) * orthogonal
        )
    )
    second_direction = (
        np.cos(theta) * ca_axis
        + np.sin(theta)
        * (
            np.cos(np.radians(240.0)) * perpendicular
            + np.sin(np.radians(240.0)) * orthogonal
        )
    )
    return (
        n_coord + N_H_BOND_LENGTH * first_direction,
        n_coord + N_H_BOND_LENGTH * second_direction,
    )


def _terminal_oxt_coord(
    ca_coord: np.ndarray,
    c_coord: np.ndarray,
    o_coord: np.ndarray,
) -> np.ndarray:
    ca_direction = _unit(ca_coord - c_coord)
    o_direction = _unit(o_coord - c_coord)
    o_perpendicular = _unit(o_direction - np.dot(o_direction, ca_direction) * ca_direction)
    angle = np.radians(CA_C_OXT_ANGLE_DEGREES)
    oxt_direction = np.cos(angle) * ca_direction - np.sin(angle) * o_perpendicular
    return c_coord + C_OXT_BOND_LENGTH * oxt_direction


def _terminal_hxt_coord(
    ca_coord: np.ndarray,
    c_coord: np.ndarray,
    o_coord: np.ndarray,
    oxt_coord: np.ndarray,
) -> np.ndarray:
    return _place_internal_coordinate_atom(
        atom_1=o_coord,
        atom_2=c_coord,
        atom_3=oxt_coord,
        bond_length=OXT_HXT_BOND_LENGTH,
        bond_angle_degrees=C_OXT_HXT_ANGLE_DEGREES,
        dihedral_degrees=180.0,
    )


def _backbone_atoms(atom_array: struc.AtomArray) -> struc.AtomArray:
    return atom_array[np.isin(atom_array.atom_name, list(BACKBONE_ATOM_NAMES))]


def _infer_nanoring_cylinder_radius(nanoring: struc.AtomArray) -> float:
    carbon_atoms = nanoring[np.char.upper(nanoring.element.astype("U2")) == "C"]
    if carbon_atoms.array_length() == 0:
        raise ValueError("nanoring must contain carbon atoms")
    radial_distances = np.linalg.norm(carbon_atoms.coord[:, :2], axis=1)
    return float(radial_distances.min())


def _generated_backbone_atoms(
    segment: SecondaryStructureSegment | TurnMotifSegment,
) -> struc.AtomArray:
    atom_array = segment.atom_array
    return atom_array[
        (atom_array.res_id != segment.bp5_residue_id)
        & np.isin(atom_array.atom_name, list(BACKBONE_ATOM_NAMES))
    ]


def _secondary_structure_direction(
    segment: SecondaryStructureSegment | TurnMotifSegment,
) -> np.ndarray:
    residue_ids = _sorted_residue_ids(segment.atom_array)
    ca_coords = np.array(
        [
            _atom_coord(_residue(segment.atom_array, residue_id), "CA")
            for residue_id in residue_ids
        ],
        dtype=float,
    )
    if ca_coords.shape[0] == 1:
        bp5_residue = _residue(segment.atom_array, segment.bp5_residue_id)
        return _unit(_atom_coord(bp5_residue, "C") - _atom_coord(bp5_residue, "N"))

    terminal_delta = ca_coords[-1] - ca_coords[0]
    if ca_coords.shape[0] == 2:
        return _unit(terminal_delta)

    centered = ca_coords - ca_coords.mean(axis=0)
    _, singular_values, right_singular_vectors = np.linalg.svd(
        centered,
        full_matrices=False,
    )
    if np.isclose(float(singular_values[0]), 0.0):
        return _unit(terminal_delta)

    direction = _unit(right_singular_vectors[0])
    if np.dot(direction, terminal_delta) < 0.0:
        direction = -direction
    return direction


def _terminal_exit_vectors(
    segment: SecondaryStructureSegment | TurnMotifSegment,
) -> tuple[np.ndarray, np.ndarray]:
    residue_ids = _sorted_residue_ids(segment.atom_array)
    bp5_residue = _residue(segment.atom_array, segment.bp5_residue_id)
    bp5_ca = _atom_coord(bp5_residue, "CA")

    n_terminal_residue = _residue(segment.atom_array, residue_ids[0])
    if residue_ids[0] == segment.bp5_residue_id:
        n_terminal_exit_vector = _unit(_atom_coord(bp5_residue, "N") - bp5_ca)
    else:
        n_terminal_exit_vector = _unit(_atom_coord(n_terminal_residue, "CA") - bp5_ca)

    c_terminal_residue = _residue(segment.atom_array, residue_ids[-1])
    if residue_ids[-1] == segment.bp5_residue_id:
        c_terminal_exit_vector = _unit(_atom_coord(bp5_residue, "C") - bp5_ca)
    else:
        c_terminal_exit_vector = _unit(_atom_coord(c_terminal_residue, "CA") - bp5_ca)

    return n_terminal_exit_vector, c_terminal_exit_vector


def _sorted_residue_ids(atom_array: struc.AtomArray) -> tuple[int, ...]:
    return tuple(sorted({int(residue_id) for residue_id in atom_array.res_id.tolist()}))


def _residue(atom_array: struc.AtomArray, residue_id: int) -> struc.AtomArray:
    return atom_array[atom_array.res_id == residue_id]


def _require_atom_names(atom_array: struc.AtomArray, atom_names: tuple[str, ...]) -> None:
    available = set(atom_array.atom_name.tolist())
    missing = sorted(set(atom_names) - available)
    if missing:
        raise ValueError(f"atom_array is missing required atom names: {missing}")


def _atom_coord(atom_array: struc.AtomArray, atom_name: str) -> np.ndarray:
    atom_indices = np.flatnonzero(atom_array.atom_name == atom_name)
    if atom_indices.size != 1:
        raise ValueError(f"expected exactly one atom named {atom_name!r}")
    return atom_array.coord[int(atom_indices[0])]


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("Cannot normalize a zero-length vector")
    return vector / norm
