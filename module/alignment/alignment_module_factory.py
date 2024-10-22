from module.alignment.merge_alignments_module import MergeAlignmentsModule
from module.alignment.only_add_unaligned_module import OnlyAddUnalignedModule


class AlignmentModuleFactory:
    @staticmethod
    def by_name(name: str):
        match name:
            case "MergeAlignmentsModule":
                return MergeAlignmentsModule
            case "OnlyAddUnalignedModule":
                return OnlyAddUnalignedModule
            case _:
                return MergeAlignmentsModule
