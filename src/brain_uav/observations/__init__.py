"""Structured observation contracts for UAV policy inputs."""

from .v2_batch import V2ObservationBatch, collate_v2_observations
from .v2_builder import build_v2_observation
from .v2_contract import (
    EGO_FEATURE_DIM,
    EGO_FEATURE_INDEX,
    EGO_FEATURE_NAMES,
    GOAL_FEATURE_DIM,
    GOAL_FEATURE_INDEX,
    GOAL_FEATURE_NAMES,
    ZONE_FEATURE_DIM,
    ZONE_FEATURE_INDEX,
    ZONE_FEATURE_NAMES,
    V2Observation,
    V2ObservationScales,
)
from .v2_relations import (
    PAIR_RELATION_FEATURE_DIM,
    PAIR_RELATION_FEATURE_INDEX,
    PAIR_RELATION_FEATURE_NAMES,
    PairRelationBuilder,
)

__all__ = [
    'EGO_FEATURE_DIM',
    'EGO_FEATURE_INDEX',
    'EGO_FEATURE_NAMES',
    'GOAL_FEATURE_DIM',
    'GOAL_FEATURE_INDEX',
    'GOAL_FEATURE_NAMES',
    'PAIR_RELATION_FEATURE_DIM',
    'PAIR_RELATION_FEATURE_INDEX',
    'PAIR_RELATION_FEATURE_NAMES',
    'PairRelationBuilder',
    'ZONE_FEATURE_DIM',
    'ZONE_FEATURE_INDEX',
    'ZONE_FEATURE_NAMES',
    'V2Observation',
    'V2ObservationBatch',
    'V2ObservationScales',
    'build_v2_observation',
    'collate_v2_observations',
]
