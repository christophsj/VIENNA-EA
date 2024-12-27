from module.alignment.higher_confidence_module import HigherConfidenceModule
from module.alignment.merge_alignments_module import MergeAlignmentsModule
from module.alignment.only_add_unaligned_module import OnlyAddUnalignedModule
from module.alignment.override_module import OverridedModule


class AlignmentModuleFactory:
    @staticmethod
    def by_name(name: str):
        match name:
            case "MergeAlignmentsModule":
                return MergeAlignmentsModule
            case "OnlyAddUnalignedModule":
                return OnlyAddUnalignedModule
            case "HigherConfidenceModule":
                return HigherConfidenceModule
            case "OverridedModule":
                return OverridedModule
            case _:
                return MergeAlignmentsModule
