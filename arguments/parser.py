import argparse
import sys
from typing import Dict, Any


class EbSynthNamespace(argparse.Namespace):
    """argparse.Namespace with the default config values and cascading-weight tracking state."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.style_file = ""
        self.style_weight = -1.0
        self.output_file = "output.png"
        self.guides = []
        self.uniformity_weight = 3500.0
        self.patch_size = 5
        self.num_pyramid_levels = -1
        self.num_search_vote_iters = 6
        self.num_patch_match_iters = 4
        self.stop_threshold = 5
        self.extra_pass_3x3 = False
        # Points at whichever -style/-guide token was parsed most recently, so a
        # following bare -weight knows what it binds to: ("style",) or ("guide", index).
        self._last_added = None


class StyleAction(argparse.Action):
    """Records -style and marks it as the current cascading-weight target."""

    def __call__(self, parser, namespace, values, option_string=None):
        namespace.style_file = values
        namespace.style_weight = -1.0
        namespace._last_added = ("style",)


class GuideAction(argparse.Action):
    """Appends a {source, target, weight} guide entry and marks it as the cascading-weight target."""

    def __call__(self, parser, namespace, values, option_string=None):
        guide_entry = {"source": values[0], "target": values[1], "weight": -1.0}
        namespace.guides.append(guide_entry)
        namespace._last_added = ("guide", len(namespace.guides) - 1)


class WeightAction(argparse.Action):
    """Applies a bare -weight to whichever -style/-guide immediately preceded it."""

    def __call__(self, parser, namespace, values, option_string=None):
        if values < 0.0:
            raise argparse.ArgumentError(self, "weights must be non-negative!")
        if namespace._last_added is None:
            raise argparse.ArgumentError(self, "at least one -style or -guide option must precede the -weight option!")

        if namespace._last_added[0] == "style":
            namespace.style_weight = values
        elif namespace._last_added[0] == "guide":
            idx = namespace._last_added[1]
            namespace.guides[idx]["weight"] = values


def parse_arguments(argv=None) -> Dict[str, Any]:
    """
    Parses CLI arguments into a plain config dict, replacing the original's tryToParseArg
    loop (ebsynth.cpp ~lines 195-304). No tensors here — output is plain Python types.

    Args:
        argv: full argv list including the program name at index 0 (e.g. sys.argv), or
              None to read sys.argv directly.

    Returns:
        dict with keys: style_file (str), style_weight (float), output_file (str),
        guides (list of {"source": str, "target": str, "weight": float}),
        uniformity_weight (float), patch_size (int), num_pyramid_levels (int),
        num_search_vote_iters (int), num_patch_match_iters (int), stop_threshold (int),
        extra_pass_3x3 (0 or 1).
    """
    parser = argparse.ArgumentParser(
        description="Ebpynth: a pure PyTorch reimplementation of ebsynth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-style", type=str, action=StyleAction, help="Path to the style painting keyframe")
    parser.add_argument("-guide", type=str, nargs=2, action=GuideAction, help="Paths to <source_guide> and <target_guide>")
    parser.add_argument("-weight", type=float, action=WeightAction, help="Weight for the preceding -style/-guide")

    parser.add_argument("-output", type=str, default="output.png", dest="output_file", help="Output filename")
    parser.add_argument("-uniformity", type=float, default=3500.0, dest="uniformity_weight", help="Spatial uniformity scaling multiplier")
    parser.add_argument("-patchsize", type=int, default=5, dest="patch_size", help="Patch Match patching radius size")
    parser.add_argument("-pyramidlevels", type=int, default=-1, dest="num_pyramid_levels", help="Image pyramid scaling depth levels")
    parser.add_argument("-searchvoteiters", type=int, default=6, dest="num_search_vote_iters", help="Search and vote execution passes per level")
    parser.add_argument("-patchmatchiters", type=int, default=4, dest="num_patch_match_iters", help="Core Patch Match randomized propagation iterations")
    parser.add_argument("-stopthreshold", type=int, default=5, dest="stop_threshold", help="Parsed for CLI compatibility only; unused by this engine")
    parser.add_argument("-extrapass3x3", action="store_true", dest="extra_pass_3x3", help="Apply a final 3x3-patch refinement pass")

    custom_namespace = EbSynthNamespace()
    try:
        target_args = argv[1:] if argv is not None else sys.argv[1:]
        args = parser.parse_args(target_args, namespace=custom_namespace)

        if args.patch_size < 3:
            parser.error("patchsize is too small!")
        if args.patch_size % 2 == 0:
            parser.error("patchsize must be an odd number!")
        if args.num_pyramid_levels < -1 or args.num_pyramid_levels == 0:
            parser.error("bad argument for -pyramidlevels!")
        if args.num_search_vote_iters < 0:
            parser.error("bad argument for -searchvoteiters!")
        if args.num_patch_match_iters < 0:
            parser.error("bad argument for -patchmatchiters!")
        if args.stop_threshold < 0:
            parser.error("bad argument for -stopthreshold!")

        config = vars(args)
        config.pop("_last_added", None)
        config["extra_pass_3x3"] = 1 if config["extra_pass_3x3"] else 0
        return config

    except SystemExit:
        sys.exit(1)


# Sandbox check (run from the repo root: python arguments/parser.py)
if __name__ == "__main__":
    mock_cli = [
        "stylize.py",
        "-style", "painting.png", "-weight", "2.0",
        "-guide", "src_flow.png", "tgt_flow.png", "-weight", "10.5",
        "-guide", "src_seg.png", "tgt_seg.png",
        "-patchsize", "7",
        "-extrapass3x3"
    ]
    res = parse_arguments(mock_cli)
    import pprint
    print("Argument parsing test passed, parsed config:")
    pprint.pprint(res)
