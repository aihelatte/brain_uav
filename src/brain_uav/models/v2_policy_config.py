"""Explicit projected-head schema and stage-local spike regularization."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class V2SpikeRegularizationConfig:
    beta_max: float = 0.0
    target_rate: float = 0.17
    ramp_start_steps: int = 25_000
    ramp_end_steps: int = 75_000

    def __post_init__(self):
        for name in ('beta_max', 'target_rate'):
            value = getattr(self, name)
            try:
                valid = type(value) in (int, float) and math.isfinite(value)
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError(f'{name} must be a finite number.')
        if self.beta_max < 0 or not 0 <= self.target_rate <= 1:
            raise ValueError('beta_max must be nonnegative and target_rate in [0, 1].')
        if (type(self.ramp_start_steps) is not int or type(self.ramp_end_steps) is not int
                or not 0 <= self.ramp_start_steps < self.ramp_end_steps):
            raise ValueError('Spike ramp requires 0 <= start < end integer stage steps.')

    def beta(self, stage_steps: int) -> float:
        if type(stage_steps) is not int or stage_steps < 0:
            raise ValueError('stage_steps must be a nonnegative integer.')
        return self.beta_max * min(1., max(0., (stage_steps - self.ramp_start_steps)
                                        / (self.ramp_end_steps - self.ramp_start_steps)))

    def validate_model(self, model: str) -> None:
        if model not in ('ann', 'snn') or (model == 'ann' and self.beta_max != 0):
            raise ValueError('Spike regularization can only be enabled for an SNN actor.')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_snapshot(cls, value):
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError('Spike regularization snapshot has missing or unknown fields.')
        return cls(**value)


def policy_head_metadata(actor):
    """Legacy actors retain their original schema and exact state-dict keys."""
    if actor.projection_dim is None:
        return {}
    return {'policy_head': {'version': 2, 'encoder_output_dim': actor.zone_set_encoder.output_dim,
                            'projection_dim': actor.projection_dim, 'hidden_dim': actor.hidden_dim}}


def projection_from_architecture(architecture):
    if 'policy_head' not in architecture:
        return None  # Explicit legacy schema: no projection, original encoder-to-head.
    head = architecture['policy_head']
    if (type(head) is not dict or set(head) != {'version', 'encoder_output_dim', 'projection_dim', 'hidden_dim'}
            or type(head['version']) is not int or head['version'] != 2):
        raise ValueError('Unsupported or incomplete policy_head schema.')
    for name in ('encoder_output_dim', 'projection_dim', 'hidden_dim'):
        if type(head[name]) is not int or head[name] <= 0:
            raise ValueError(f'policy_head {name} must be a positive integer.')
    try:
        consistent = (head['hidden_dim'] == architecture['actor_hidden_dim']
                      and head['encoder_output_dim'] == 2 * architecture['encoder_config']['hidden_dim'])
    except (KeyError, TypeError) as exc:
        raise ValueError('Incomplete policy/encoder architecture.') from exc
    if not consistent:
        raise ValueError('policy_head dimensions contradict encoder/actor metadata.')
    return head['projection_dim']


def validate_policy_request(actor, projection_dim=None, hidden_dim=None):
    validate_policy_dimensions(projection_dim, hidden_dim)
    for name, requested, actual in (('actor_projection_dim', projection_dim, actor.projection_dim),
                                     ('actor_hidden_dim', hidden_dim, actor.hidden_dim)):
        if requested is not None and (type(requested) is not int or requested <= 0 or requested != actual):
            raise ValueError(f'{name} does not match the strictly loaded initialization actor.')


def validate_policy_dimensions(projection_dim=None, hidden_dim=None):
    for name, value in (('actor_projection_dim', projection_dim), ('actor_hidden_dim', hidden_dim)):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f'{name} must be a positive integer.')


def add_policy_training_arguments(parser):
    parser.add_argument('--actor-projection-dim', type=int, default=None)
    parser.add_argument('--actor-hidden-dim', type=int, default=None)
    parser.add_argument('--spike-beta-max', type=float, default=None)
    parser.add_argument('--spike-target-rate', type=float, default=.17)
    parser.add_argument('--spike-ramp-start-steps', type=int, default=25_000)
    parser.add_argument('--spike-ramp-end-steps', type=int, default=75_000)
    parser.add_argument('--actor-rl-scale-alpha', type=float, default=2.5)


def spike_config_from_arguments(args):
    if args.spike_beta_max is None:
        if (args.spike_target_rate, args.spike_ramp_start_steps, args.spike_ramp_end_steps) != (.17, 25_000, 75_000):
            raise ValueError('Explicit spike schedule requires --spike-beta-max.')
        return None  # Stage continuation inherits the predecessor; easy disables it.
    result = V2SpikeRegularizationConfig(args.spike_beta_max, args.spike_target_rate,
                                         args.spike_ramp_start_steps, args.spike_ramp_end_steps)
    result.validate_model(args.model)
    return result
