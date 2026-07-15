import argparse
import sys
from typing import Dict, Any

class EbSynthNamespace(argparse.Namespace):
    """Custom tracker namespace maintaining order-dependent state for cascading weights."""
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
        self._last_added = None  # State anchor tracking immediately preceding asset: ('style',) or ('guide', index)

class StyleAction(argparse.Action):
    """Custom action triggered instantly when '-style' token is encountered."""
    def __call__(self, parser, namespace, values, option_string=None):
        namespace.style_file = values
        namespace.style_weight = -1.0
        namespace._last_added = ("style",)

class GuideAction(argparse.Action):
    """Custom action triggered instantly when '-guide' token pair is encountered."""
    def __call__(self, parser, namespace, values, option_string=None):
        guide_entry = {"source": values[0], "target": values[1], "weight": -1.0}
        namespace.guides.append(guide_entry)
        namespace._last_added = ("guide", len(namespace.guides) - 1)

class WeightAction(argparse.Action):
    """Custom action evaluating contextual binding for cascading weights."""
    def __call__(self, parser, namespace, values, option_string=None):
        if values < 0.0:
            raise argparse.ArgumentError(self, "weights must be non-negative!")
        if namespace._last_added is None:
            raise argparse.ArgumentError(self, "at least one -style or -guide option must precede the -weight option!")
        
        # Dynamically route the weight value based on chronological invocation order
        if namespace._last_added[0] == "style":
            namespace.style_weight = values
        elif namespace._last_added[0] == "guide":
            idx = namespace._last_added[1]
            namespace.guides[idx]["weight"] = values


def parse_arguments(argv=None) -> Dict[str, Any]:
    """
    Modernized argument utility replacing legacy C++ loop scanners with standard argparse.
    Handles type checking, out-of-bounds validation, and help menus automatically.
    """
    parser = argparse.ArgumentParser(
        description="EbSynth High-Performance GPU Pipeline (PyTorch Native Edition)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 🎯 Register core structural assets with specialized sequential custom actions
    parser.add_argument("-style", type=str, action=StyleAction, help="Path to the style painting keyframe")
    parser.add_argument("-guide", type=str, nargs=2, action=GuideAction, help="Paths to <source_guide> and <target_guide>")
    parser.add_argument("-weight", type=float, action=WeightAction, help="Contextual weight multiplier for preceding asset")
    
    # 🎯 Register standard hyper-parameters with built-in automatic type conversion
    parser.add_argument("-output", type=str, default="output.png", dest="output_file", help="Output filename")
    parser.add_argument("-uniformity", type=float, default=3500.0, dest="uniformity_weight", help="Spatial uniformity scaling multiplier")
    parser.add_argument("-patchsize", type=int, default=5, dest="patch_size", help="Patch Match patching radius size")
    parser.add_argument("-pyramidlevels", type=int, default=-1, dest="num_pyramid_levels", help="Image pyramid scaling depth levels")
    parser.add_argument("-searchvoteiters", type=int, default=6, dest="num_search_vote_iters", help="Search and vote execution passes per level")
    parser.add_argument("-patchmatchiters", type=int, default=4, dest="num_patch_match_iters", help="Core Patch Match randomized propagation iterations")
    parser.add_argument("-stopthreshold", type=int, default=5, dest="stop_threshold", help="Early termination energy break limit")
    parser.add_argument("-extrapass3x3", action="store_true", dest="extra_pass_3x3", help="Apply secondary 3x3 median noise cleanup pass")

    # Bind parsing operation to our custom unified state tracking namespace
    custom_namespace = EbSynthNamespace()
    try:
        # If argv is passed from custom array list, ignore program name token [0]
        target_args = argv[1:] if argv is not None else sys.argv[1:]
        args = parser.parse_args(target_args, namespace=custom_namespace)
        
        # Domain Security Gates: Domain-specific mathematical integrity checks
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

        # Flatten namespace down to dictionary config map
        config = vars(args)
        config.pop("_last_added", None) # Evict internal state memory prior to pipeline ingestion
        
        # Cast boolean switch into absolute 0 or 1 integer flag to comply with upstream C++ expectations
        config["extra_pass_3x3"] = 1 if config["extra_pass_3x3"] else 0
        
        return config

    except SystemExit:
        # Gracefully handle default argparse termination signals
        sys.exit(1)


# Sandbox validation grid execution
if __name__ == "__main__":
    mock_cli = [
        "pipeline.py",
        "-style", "painting.png", "-weight", "2.0",
        "-guide", "src_flow.png", "tgt_flow.png", "-weight", "10.5",
        "-guide", "src_seg.png", "tgt_seg.png",
        "-patchsize", "7",
        "-extrapass3x3"
    ]
    res = parse_arguments(mock_cli)
    import pprint
    print("🏆 Argument Parsing Engine Test Passed! Output map configuration:")
    pprint.pprint(res)