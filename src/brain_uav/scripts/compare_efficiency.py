"""Compare ANN and SNN efficiency summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

AC_ENERGY_PJ = 0.9
MAC_ENERGY_PJ = 4.6


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _number(payload: dict, *keys: str) -> float | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _time_requirement(value: float | None) -> bool | None:
    return None if value is None else value <= 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare ANN/SNN efficiency summaries.')
    parser.add_argument('--ann', type=Path, required=True)
    parser.add_argument('--snn', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('compare_efficiency.json'))
    args = parser.parse_args()

    ann = _load(args.ann)
    snn = _load(args.snn)

    ann_params = _number(ann, 'param_count')
    snn_params = _number(snn, 'param_count')
    ann_macs = _number(ann, 'ann_macs', 'dense_theoretical_macs')
    snn_syops_total = _number(snn, 'syops_total')
    snn_acs = _number(snn, 'snn_acs')
    snn_macs = _number(snn, 'snn_macs')
    snn_spike_aware_ops = _number(snn, 'snn_spike_aware_ops', 'snn_syops')
    if snn_spike_aware_ops is None and snn_acs is not None and snn_macs is not None:
        snn_spike_aware_ops = snn_acs + snn_macs
    snn_energy = _number(snn, 'snn_energy_pj', 'syops_energy')
    if snn_energy is None and snn_acs is not None and snn_macs is not None:
        snn_energy = snn_acs * AC_ENERGY_PJ + snn_macs * MAC_ENERGY_PJ
    ann_energy = _number(ann, 'ann_energy_pj', 'ann_energy', 'energy', 'syops_energy')
    if ann_energy is None and ann_macs is not None:
        ann_energy = ann_macs * MAC_ENERGY_PJ
    ann_1000s_est = _number(ann, 'estimated_1000s_planning_time_s')
    snn_1000s_est = _number(snn, 'estimated_1000s_planning_time_s')
    ann_1000s_meas = _number(ann, 'measured_1000s_planning_time_s')
    snn_1000s_meas = _number(snn, 'measured_1000s_planning_time_s')

    mac_reduction_ratio = None
    if ann_macs is not None and ann_macs > 0.0 and snn_macs is not None:
        mac_reduction_ratio = 1.0 - (snn_macs / ann_macs)

    spike_aware_ops_reduction_ratio = None
    if ann_macs is not None and ann_macs > 0.0 and snn_spike_aware_ops is not None:
        spike_aware_ops_reduction_ratio = 1.0 - (snn_spike_aware_ops / ann_macs)

    raw_syops_reduction_ratio = None
    if ann_macs is not None and ann_macs > 0.0 and snn_syops_total is not None:
        raw_syops_reduction_ratio = 1.0 - (snn_syops_total / ann_macs)

    energy_reduction_ratio = None
    if ann_energy is not None and ann_energy > 0.0 and snn_energy is not None:
        energy_reduction_ratio = 1.0 - (snn_energy / ann_energy)

    report = {
        'ann_param_count': ann_params,
        'snn_param_count': snn_params,
        'param_count_close': (
            None if ann_params is None or snn_params is None or ann_params <= 0.0
            else abs(ann_params - snn_params) <= 0.01 * ann_params
        ),
        'ann_macs': ann_macs,
        'snn_macs': snn_macs,
        'snn_acs': snn_acs,
        'snn_spike_aware_ops': snn_spike_aware_ops,
        'syops_total': snn_syops_total,
        'ann_energy_pj': ann_energy,
        'snn_energy_pj': snn_energy,
        'mac_reduction_ratio': mac_reduction_ratio,
        'meets_mac_reduction_requirement': None if mac_reduction_ratio is None else mac_reduction_ratio >= 0.5,
        'energy_reduction_ratio': energy_reduction_ratio,
        'meets_energy_reduction_50pct': None if energy_reduction_ratio is None else energy_reduction_ratio >= 0.5,
        'raw_syops_reduction_ratio': raw_syops_reduction_ratio,
        'spike_aware_ops_reduction_ratio': spike_aware_ops_reduction_ratio,
        'ann_estimated_1000s_planning_time_s': ann_1000s_est,
        'snn_estimated_1000s_planning_time_s': snn_1000s_est,
        'ann_measured_1000s_planning_time_s': ann_1000s_meas,
        'snn_measured_1000s_planning_time_s': snn_1000s_meas,
        'ann_meets_time_requirement': _time_requirement(ann_1000s_meas),
        'snn_meets_time_requirement': _time_requirement(snn_1000s_meas),
        'ann_macs_method': ann.get('dense_macs_method') or ann.get('macs_counting_method'),
        'snn_macs_method': snn.get('syops_method') or snn.get('macs_counting_method'),
        'comparison_notes': (
            'Hard MAC requirement uses mac_reduction_ratio = 1 - snn_macs / ann_macs; SNN ACs are excluded from '
            'the MAC-reduction numerator. Energy reduction uses ANN MAC energy and SNN AC/MAC mixed energy. '
            'Time requirement uses measured_1000s_planning_time_s <= 1.0.'
        ),
    }

    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f"ANN 1000s planning time <= 1s: {report['ann_meets_time_requirement']}")
    print(f"SNN 1000s planning time <= 1s: {report['snn_meets_time_requirement']}")
    print(f"SNN MAC reduction >= 50%: {report['meets_mac_reduction_requirement']}")
    print(f"Saved compare report to {args.output}")


if __name__ == '__main__':
    main()
