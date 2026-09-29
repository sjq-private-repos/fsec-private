from collections import defaultdict

import numpy as np
from pyscf.lib import logger
from pyscf.pbc.tools import get_monkhorst_pack_size

from fsec.singularity_subtraction.grids import minimum_image


class TimingProfile:
    """Accumulate PySCF CPU and wall-clock timings for repeated regions."""

    def __init__(self):
        self._times = defaultdict(lambda: [0.0, 0.0])

    @staticmethod
    def start():
        return logger.process_clock(), logger.perf_counter()

    def stop(self, label, start):
        cpu = logger.process_clock() - start[0]
        wall = logger.perf_counter() - start[1]
        self._times[label][0] += cpu
        self._times[label][1] += wall
        return cpu, wall

    def summary(self, total_start):
        total_cpu = logger.process_clock() - total_start[0]
        total_wall = logger.perf_counter() - total_start[1]
        summary = {
            label: {"cpu": values[0], "wall": values[1]}
            for label, values in self._times.items()
        }
        summary["total"] = {"cpu": total_cpu, "wall": total_wall}
        return summary

    @staticmethod
    def log_summary(log, summary, title="build_structure_factor"):
        total = summary["total"]
        log.note(
            "%s CPU %.2f sec, wall %.2f sec",
            title, total["cpu"], total["wall"],
        )
        for label, values in sorted(
                ((key, value) for key, value in summary.items() if key != "total"),
                key=lambda item: item[1]["wall"], reverse=True):
            cpu_fraction = 100.0 * values["cpu"] / total["cpu"] if total["cpu"] else 0.0
            wall_fraction = 100.0 * values["wall"] / total["wall"] if total["wall"] else 0.0
            log.note(
                "  %-36s CPU %9.2f sec (%5.1f%%), wall %9.2f sec (%5.1f%%)",
                label, values["cpu"], cpu_fraction,
                values["wall"], wall_fraction,
            )


def normalize_line_sampling_decay_components(components, supported_components):
    if components is None:
        return ()
    if isinstance(components, str):
        components = (components,)
    components = tuple(components)
    supported_components = set(supported_components)
    unsupported = sorted(set(components) - supported_components)
    if unsupported:
        raise NotImplementedError(
            "Line-sampling decay filtering is not implemented for "
            f"{unsupported}. Supported components: {sorted(supported_components)}"
        )
    return components


def filter_line_sampling_segments(segments, keep_mask):
    if segments is None:
        return None
    keep_mask = np.asarray(keep_mask, dtype=bool)
    old_to_new = np.full(keep_mask.shape[0], -1, dtype=int)
    old_to_new[keep_mask] = np.arange(np.count_nonzero(keep_mask), dtype=int)
    filtered_segments = []
    for segment in segments:
        old_indices = np.asarray(segment["indices"], dtype=int)
        valid = old_indices[old_indices < keep_mask.shape[0]]
        valid = valid[keep_mask[valid]]
        new_indices = old_to_new[valid]
        filtered_segment = dict(segment)
        filtered_segment["indices"] = new_indices
        filtered_segments.append(filtered_segment)
    return filtered_segments


def line_sampling_decay_decision(value, norm, state, min_fraction, consecutive_below, power):
    contribution = np.inf if norm <= 1e-8 else abs(value) / norm**power
    if state["reference"] is None:
        state["reference"] = contribution
        state["threshold"] = min_fraction * contribution
        state["below_count"] = 0
        return True

    if contribution < state["threshold"]:
        state["below_count"] += 1
        if state["below_count"] >= consecutive_below:
            state["stopped"] = True
        return False

    state["below_count"] = 0
    return True


def make_line_sampling_decay_state(qG_full, segments, min_fraction,
                                   consecutive_below, power=4):
    if segments is None or min_fraction is None or min_fraction <= 0:
        return None
    if consecutive_below is None or int(consecutive_below) < 1:
        raise ValueError("line_sampling_decay_consecutive_below must be at least 1")

    index_to_line = {}
    line_states = {}
    for segment in segments:
        line_index = segment["line_index"]
        indices = np.asarray(segment["indices"], dtype=int)
        if indices.size == 0:
            continue
        line_states[line_index] = {
            "B_index": segment.get("B_index", line_index),
            "step_vector": np.asarray(segment["step_vector"]),
            "q_min": None if segment.get("q_min") is None else np.asarray(segment["q_min"]),
            "reference": None,
            "threshold": None,
            "below_count": 0,
            "stopped": False,
            "stop_event": None,
        }
        for index in indices:
            if index < qG_full.shape[0]:
                index_to_line[int(index)] = line_index

    return {
        "index_to_line": index_to_line,
        "line_states": line_states,
        "min_fraction": float(min_fraction),
        "consecutive_below": int(consecutive_below),
        "power": power,
        "stop_events": [],
    }


def should_compute_line_sample(index, decay_state):
    if decay_state is None:
        return True
    line_index = decay_state["index_to_line"].get(int(index))
    if line_index is None:
        return True
    return not decay_state["line_states"][line_index]["stopped"]


def update_line_sampling_decay_mask(mask, index, value, norm, decay_state, qGpt=None):
    if decay_state is None:
        mask[index] = True
        return
    line_index = decay_state["index_to_line"].get(int(index))
    if line_index is None:
        mask[index] = True
        return
    line_state = decay_state["line_states"][line_index]
    was_stopped = line_state["stopped"]
    keep = line_sampling_decay_decision(
        value,
        norm,
        line_state,
        decay_state["min_fraction"],
        decay_state["consecutive_below"],
        decay_state["power"],
    )
    mask[index] = keep
    if line_state["stopped"] and not was_stopped:
        contribution = np.inf if norm <= 1e-8 else abs(value) / norm**decay_state["power"]
        stop_event = {
            "line_index": line_index,
            "B_index": line_state["B_index"],
            "step_vector": line_state["step_vector"].copy(),
            "q_min": None if line_state["q_min"] is None else line_state["q_min"].copy(),
            "qG_index": int(index),
            "qG": None if qGpt is None else np.asarray(qGpt).copy(),
            "qG_norm": float(norm),
            "value": value,
            "normalized_contribution": float(contribution),
            "threshold": float(line_state["threshold"]),
            "reference": float(line_state["reference"]),
            "min_fraction": decay_state["min_fraction"],
            "consecutive_below": decay_state["consecutive_below"],
        }
        line_state["stop_event"] = stop_event
        decay_state["stop_events"].append(stop_event)


def build_uKpts(kmf, kpts, mo_coeff_kpts, NsCell=None, rptGrid3D=None, nbands=None):
    # Setup constants
    NsCell = np.array(kmf.cell.mesh) if NsCell is None else NsCell
    nbands = kmf.cell.tot_electrons() // 2 if nbands is None else nbands
    nks = get_monkhorst_pack_size(kmf.cell, kpts)
    Nk = np.prod(nks)

    # Setup real space grid points
    if rptGrid3D is None:
        Lvec_real = kmf.cell.lattice_vectors()
        L_delta = Lvec_real / NsCell[:, None]
        xv, yv, zv = np.meshgrid(
            np.arange(NsCell[0]),
            np.arange(NsCell[1]),
            np.arange(NsCell[2]),
            indexing='ij',
        )
        mesh_idx = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1), zv.reshape(-1, 1)])
        rptGrid3D = mesh_idx @ L_delta

    assert rptGrid3D.shape[1] == 3, "build_uKpts: rptGrid3D should be a 3D array"
    nG = rptGrid3D.shape[0]

    # Evaluate the atomic orbitals at the real space grid points
    kGrid = minimum_image(kmf.cell, kpts)
    aoval = kmf.cell.pbc_eval_gto("GTOval_sph", coords=rptGrid3D, kpts=kpts)

    # Compute uKpts
    exp_part = np.exp(-1j * (rptGrid3D @ kGrid.T)).T
    utmp = aoval @ np.array(mo_coeff_kpts)[:, :, :nbands]
    utmp = utmp.transpose(0, 2, 1)
    uKpts = exp_part[:, None, :] * utmp
    return uKpts
