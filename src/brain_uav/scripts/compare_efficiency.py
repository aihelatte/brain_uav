"""Compare ANN and SNN efficiency summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

AC_ENERGY_PJ = 0.9
MAC_ENERGY_PJ = 4.6


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare ANN/SNN efficiency summaries.')
    parser.add_argument('--ann', type=Path, required=True)
    parser.add_argument('--snn', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('compare_efficiency.json'))
    args = parser.parse_args()

    ann = _load(args.ann)
    snn = _load(args.snn)

    ann_params = ann.get('param_count')
    snn_params = snn.get('param_count')
    ann_macs = ann.get('ann_macs') or ann.get('dense_theoretical_macs')
    snn_syops_total = snn.get('syops_total')
    snn_acs = snn.get('snn_acs')
    snn_macs = snn.get('snn_macs')
    snn_spike_aware_ops = snn.get('snn_spike_aware_ops') or snn.get('snn_syops')
    if snn_spike_aware_ops is None and snn_acs is not None and snn_macs is not None:
        snn_spike_aware_ops = float(snn_acs) + float(snn_macs)
    snn_energy = snn.get('snn_energy_pj') or snn.get('syops_energy')
    if snn_energy is None and snn_acs is not None and snn_macs is not None:
        snn_energy = float(snn_acs) * AC_ENERGY_PJ + float(snn_macs) * MAC_ENERGY_PJ
    ann_energy = ann.get('ann_energy_pj') or ann.get('ann_energy') or ann.get('energy') or ann.get('syops_energy')
    if ann_energy is None and ann_macs is not None:
        ann_energy = float(ann_macs) * MAC_ENERGY_PJ
    ann_1000s_est = ann.get('estimated_1000s_planning_time_s')
    snn_1000s_est = snn.get('estimated_1000s_planning_time_s')
    ann_1000s_meas = ann.get('measured_1000s_planning_time_s')
    snn_1000s_meas = snn.get('measured_1000s_planning_time_s')

    spike_aware_ops_reduction_ratio = None
    if ann_macs is not None and snn_spike_aware_ops is not None:
        spike_aware_ops_reduction_ratio = 1.0 - (snn_spike_aware_ops / ann_macs)

    raw_syops_reduction_ratio = None
    if ann_macs is not None and snn_syops_total is not None:
        raw_syops_reduction_ratio = 1.0 - (snn_syops_total / ann_macs)

    energy_reduction_ratio = None
    if ann_energy is not None and snn_energy is not None:
        energy_reduction_ratio = 1.0 - (snn_energy / ann_energy)

    report = {
        'ann_param_count': ann_params,
        'snn_param_count': snn_params,
        'param_count_close': None if ann_params is None or snn_params is None else abs(ann_params - snn_params) <= 0.01 * ann_params,
        'ann_dense_theoretical_macs': ann_macs,
        'snn_syops_total_raw': snn_syops_total,
        'snn_spike_aware_ops': snn_spike_aware_ops,
        'snn_acs': snn_acs,
        'snn_macs': snn_macs,
        'ann_energy_pj': ann_energy,
        'snn_energy_pj': snn_energy,
        'ann_macs': ann_macs,
        'raw_syops_reduction_ratio': raw_syops_reduction_ratio,
        'spike_aware_ops_reduction_ratio': spike_aware_ops_reduction_ratio,
        'mac_reduction_ratio': spike_aware_ops_reduction_ratio,
        'meets_macs_reduction_50pct': None if spike_aware_ops_reduction_ratio is None else spike_aware_ops_reduction_ratio >= 0.5,
        'energy_reduction_ratio': energy_reduction_ratio,
        'estimated_energy_ratio': energy_reduction_ratio,
        'ann_estimated_1000s_planning_time_s': ann_1000s_est,
        'snn_estimated_1000s_planning_time_s': snn_1000s_est,
        'ann_measured_1000s_planning_time_s': ann_1000s_meas,
        'snn_measured_1000s_planning_time_s': snn_1000s_meas,
        'ann_meets_time_target': None if ann_1000s_meas is None else ann_1000s_meas <= 1.0,
        'snn_meets_time_target': None if snn_1000s_meas is None else snn_1000s_meas <= 1.0,
        'ann_macs_method': ann.get('dense_macs_method') or ann.get('macs_counting_method'),
        'snn_macs_method': snn.get('syops_method') or snn.get('macs_counting_method'),
        'comparison_notes': (
            'Time target uses measured_1000s_planning_time_s. Operation reduction uses ANN dense_theoretical_macs '
            'vs SNN spike-aware ops (ACs + MACs), not raw syops_total. Energy estimate uses AC=0.9 pJ and MAC=4.6 pJ.'
        ),
    }

    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f"ANN 1000s planning time <= 1s: {report['ann_meets_time_target']}")
    print(f"SNN 1000s planning time <= 1s: {report['snn_meets_time_target']}")
    print(f"SNN MAC reduction >= 50%: {report['meets_macs_reduction_50pct']}")
    print(f"Saved compare report to {args.output}")


if __name__ == '__main__':
    main()
