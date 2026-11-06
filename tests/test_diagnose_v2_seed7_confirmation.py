import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from brain_uav.scripts import diagnose_v2_seed7_confirmation as d
from test_diagnose_v2_online_confirmation import dump
from test_diagnose_v2_online_confirmation import scenes,payload


class Seed7Confirmation(unittest.TestCase):
    def args(self,root):
        return d.parser().parse_args([v for k in ('candidate-lock','batch','sealed-confirmation','validation-pool',
            'snn-bc','output-dir','historical-repo') for v in ('--'+k,str(root/k))])

    def test_default_plan_no_side_effects(self):
        with TemporaryDirectory() as tmp:
            a=self.args(Path(tmp))
            with patch.object(d.base.cal.base,'load_models',side_effect=AssertionError), \
                 patch.object(d,'provenance',side_effect=AssertionError),patch.object(d.torch.cuda,'init',side_effect=AssertionError):
                r=d.execute(a)
            self.assertEqual(r['candidates'],list(d.IDS));self.assertEqual(r['independent_flights'],120)
            self.assertEqual(r['branch_limit'],216);self.assertEqual(list(Path(tmp).iterdir()),[])

    def fixture(self,a):
        jobs=[j for j in d.base.research.matrix(7) if j['id'] in d.IDS]
        dump(a.batch/'manifest.json',dict(plan=dict(jobs=jobs)))
        rows=[]
        for j in jobs:
            path=a.batch/j['id']/'other/final.pt';path.parent.mkdir(parents=True);path.write_bytes(j['id'].encode())
            h=d.base.file_sha256(path)
            rows.append(dict(id=j['id'],model='snn',arm=j['arm'],seed=7,requested_step=400000,actual_step=400000,
                local_checkpoint=str(path),sha256=h))
            review=a.batch/j['id']/'review'
            dump(review/'summary.json',dict(job=j,execution_completed=True,actual_step=400000))
            dump(review/'progress.json',dict(state='completed_budget',stage_steps=400000,arm=j['arm']))
            dump(review/'process.json',dict(exit_code=0))
            dump(review/'snapshot_400000_400000.json',dict(sha256=h,requested_step=400000,actual_step=400000,
                pool_digest=d.base.research.DIGEST,BC_sha256=d.base.research.BC['snn'][1]))
        dump(a.candidate_lock,dict(candidates=rows));return rows

    def test_three_locked_candidates_seed_steps_sidecars_and_strict_load(self):
        with TemporaryDirectory() as tmp:
            a=self.args(Path(tmp));rows=self.fixture(a)
            def raw(path,**kw):
                r=next(r for r in rows if r['local_checkpoint']==str(path))
                return dict(arm=r['arm'],requested_step=400000,actual_step=400000,snapshot=dict(formal_config=dict(seed=7)))
            with patch.object(d.torch,'load',side_effect=raw),patch.object(d.base,'validate_snapshot') as validate:
                self.assertEqual(len(d.candidate_inputs(a,d.base.Inputs(),load=True)),3)
                self.assertEqual(validate.call_count,3);self.assertEqual(validate.call_args.kwargs['expected_model_type'],'snn')
                for key,value in (('seed',23),('actual_step',399999),('sha256','bad')):
                    changed=json.loads(json.dumps(rows));changed[0][key]=value;dump(a.candidate_lock,dict(candidates=changed))
                    with self.subTest(key=key),self.assertRaises(ValueError):d.candidate_inputs(a,d.base.Inputs(),load=True)
                dump(a.candidate_lock,dict(candidates=rows))
                with patch.object(d.torch,'load',return_value=dict(arm='min',requested_step=400000,actual_step=400000,
                        snapshot=dict(formal_config=dict(seed=23)))):
                    with self.assertRaisesRegex(ValueError,'seed/arm'):d.candidate_inputs(a,d.base.Inputs(),load=True)
                (a.batch/d.IDS[0]/'review/process.json').unlink()
                with self.assertRaises(FileNotFoundError):d.candidate_inputs(a,d.base.Inputs(),load=True)

    def test_failure_summary_and_no_generation_or_pool_substitution(self):
        with TemporaryDirectory() as tmp:
            a=self.args(Path(tmp));a.launch=True
            with patch.object(d,'provenance',side_effect=ValueError('sealed pool mismatch')), \
                 patch.object(d.base,'select_scenes',side_effect=AssertionError('must not generate')):
                with self.assertRaises(ValueError):d.execute(a)
            r=d.base.read(a.output_dir/'review/summary.json')
            self.assertFalse(r['execution_completed']);self.assertIn('sealed pool mismatch',r['error'])

    def test_incomplete_reuse_rejected_without_model_or_flight(self):
        with TemporaryDirectory() as tmp:
            a=self.args(Path(tmp));a.launch=True
            dump(a.output_dir/'review/summary.json',dict(execution_completed=False))
            with patch.object(d.base.cal.base,'load_models',side_effect=AssertionError),patch.object(d.base.cal.base,'fly',side_effect=AssertionError):
                with self.assertRaises(ValueError):d.execute(a)

    def test_real_log_coverage_fingerprints_and_sealed_scene_collision(self):
        with TemporaryDirectory() as tmp:
            batch=Path(tmp);excluded=set()
            for job in d.IDS:
                p=batch/job/'other/training_scenes.jsonl';p.parent.mkdir(parents=True)
                p.write_text('\n'.join(json.dumps(dict(start_step=s,payload=payload(100+s),
                    physical_sha256=d.base.offline.scene_key(payload(100+s)))) for s in (0,2)),encoding='utf-8')
                ep=batch/job/'other/metrics_reports/episodes.jsonl';ep.parent.mkdir()
                ep.write_text(json.dumps(dict(stage_steps=2)),encoding='utf-8')
                dump(batch/job/'review/summary.json',dict(execution_completed=True,actual_step=400000,episodes=[dict(stage_steps=2)]))
            self.assertEqual(len(d.training_exclusions(batch,d.base.Inputs(),excluded)),3)
            d.base.check_scenes(scenes(),excluded)
            excluded.add(d.base.offline.scene_key(scenes()[0]['payload']))
            with self.assertRaises(ValueError):d.base.check_scenes(scenes(),excluded)
            p=batch/d.IDS[0]/'other/training_scenes.jsonl';original=p.read_text()
            p.write_text(original.splitlines()[0],encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'incomplete'):d.training_exclusions(batch,d.base.Inputs(),set())
            p.write_text(original.replace('"start_step": 2','"start_step": 3'),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'incomplete'):d.training_exclusions(batch,d.base.Inputs(),set())
            p.write_text(original.replace('"physical_sha256": "','"physical_sha256": "wrong'),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'fingerprint'):d.training_exclusions(batch,d.base.Inputs(),set())
            p.write_text(original,encoding='utf-8')
            ep=batch/d.IDS[0]/'other/metrics_reports/episodes.jsonl'
            summary=batch/d.IDS[0]/'review/summary.json'
            dump(summary,dict(execution_completed=True,actual_step=400000,episodes=[dict(stage_steps=2),dict(stage_steps=4)]))
            # Both logs consistently truncated still disagree with the complete summary.
            with self.assertRaisesRegex(ValueError,'incomplete'):d.training_exclusions(batch,d.base.Inputs(),set())
            ep.write_text(json.dumps(dict(stage_steps=400000)),encoding='utf-8')
            dump(summary,dict(execution_completed=True,actual_step=400000,episodes=[dict(stage_steps=400000)]))
            p.write_text(original.splitlines()[0],encoding='utf-8')
            d.training_exclusions(batch,d.base.Inputs(),set())
            ep.write_text(json.dumps(dict(stage_steps=400001)),encoding='utf-8')
            dump(summary,dict(execution_completed=True,actual_step=400000,episodes=[dict(stage_steps=400001)]))
            with self.assertRaisesRegex(ValueError,'incomplete'):d.training_exclusions(batch,d.base.Inputs(),set())

    def test_shared_helpers_remain_real_and_old_contract_unchanged(self):
        self.assertEqual(len(d.base.IDS),6);self.assertNotIn('snn_fixed_own',d.base.IDS)
        # Seed is an explicit argument to strict BC preparation, not a global rewrite.
        data=dict(device='cpu');prepared=SimpleNamespace(bc_initialization=SimpleNamespace(actor=__import__('torch').nn.Linear(2,2)))
        with patch('brain_uav.trainers.v2_formal_training.prepare_v2_stage_initialization',return_value=prepared) as p:
            d.base.bc_data(data,'snn',Path('best'),seed=7)
        self.assertEqual(p.call_args.args[0].seed,7)

    def test_actual_execute_final_seal_error_keeps_false(self):
        import torch
        from contextlib import ExitStack
        with TemporaryDirectory() as tmp,ExitStack() as stack:
            a=self.args(Path(tmp));a.launch=True
            source=a.sealed_confirmation/'other/independent_pool.json';dump(source,{'sealed':40})
            rows=[dict(id=j,model='snn',path=str(Path(tmp)/(j+'.pt')),sha256='fixture') for j in d.IDS]
            models=[torch.nn.Linear(2,1).eval().requires_grad_(False) for _ in range(3)]
            data=dict(actor=models[0],critics=models[1:],device='cpu',checkpoint_hash='fixture',
                checkpoint_payload={'formal_config':{'seed':7}})
            pool=SimpleNamespace(scenarios=[])
            stack.enter_context(patch.object(d,'provenance',return_value=(rows,None,pool,{'excluded'},[])))
            stack.enter_context(patch.object(d,'candidate_inputs',return_value=rows))
            stack.enter_context(patch.object(d.base.cal.base,'load_models',return_value=data))
            stack.enter_context(patch.object(d.base,'validate_loaded'))
            stack.enter_context(patch.object(d.base,'bc_data',return_value=data))
            stack.enter_context(patch.object(d.base,'anchors',return_value=[]))
            stack.enter_context(patch.object(d.base,'flight_metrics',return_value=dict(goal_rate=0.,outcomes={},flights=[])))
            def seal(data,bc,shared,directory):
                for i in range(12):dump(directory/f'plan_{i:02d}.json',dict(index=i))
                return []
            def branches(*args):
                self.assertEqual(len(list((a.output_dir/'other').glob('*/development_probes/plan_*.json'))),36)
                return []
            stack.enter_context(patch.object(d.base,'seal_probe_plans',side_effect=seal))
            branch_calls=stack.enter_context(patch.object(d.base,'run_probe_branches',side_effect=branches))
            stack.enter_context(patch.object(d.base.Inputs,'check'))
            a.validation_pool.write_bytes(b'old pool')
            original=d.base.file_sha256
            def fail(p):
                if Path(p)==a.output_dir/'other/snn_acc_001/probe_results.json':raise OSError('seal fault')
                return original(p)
            with patch.object(d.base,'file_sha256',side_effect=fail):
                with self.assertRaisesRegex(OSError,'seal fault'):d.execute(a)
            r=d.base.read(a.output_dir/'review/summary.json')
            self.assertFalse(r['execution_completed']);self.assertIn('seal fault',r['error'])
            self.assertEqual(branch_calls.call_count,3)


if __name__=='__main__':unittest.main()
