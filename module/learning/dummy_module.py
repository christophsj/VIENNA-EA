from module.module import AlignmentState, Module, entity_alignment
from objects.KG import KG


class DummyModule(Module):

    def step(self, kg1: KG, kg2: KG, state: AlignmentState) -> AlignmentState:
        return state
