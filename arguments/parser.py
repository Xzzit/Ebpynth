import argparse
import sys
from typing import Dict, Any


def parse_arguments(argv=None) -> Dict[str, Any]:
    """
    Parses CLI arguments into a plain config dict, replacing the original's tryToParseArg
    loop (ebsynth.cpp ~lines 195-304). Weight is inline on -style/-guide (nargs="+", the
    trailing token parsed as a float when present) instead of a separate cascading -weight
    flag, so there's no parser state tracking "which flag was declared last." No tensors
    here — output is plain Python types.

    Args:
        argv: full argv list including the program name at index 0 (e.g. sys.argv), or
              None to read sys.argv directly.

    Returns:
        dict with keys: style_file (str), style_weight (float, -1.0 if unset),
        output_file (str), guides (list of {"source": str, "target": str,
        "weight": float}, -1.0 if unset), uniformity_weight (float), patch_size (int),
        num_pyramid_levels (int), num_search_vote_iters (int), num_patch_match_iters (int),
        stop_threshold (int), extra_pass_3x3 (0 or 1).
    """
    parser = argparse.ArgumentParser(
        description="Ebpynth: a pure PyTorch reimplementation of ebsynth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-style", nargs="+", required=True, metavar=("PATH", "WEIGHT"),
                        help="Path to the style painting keyframe, optionally followed by its weight (default 1.0)")
    parser.add_argument("-guide", dest="guides", action="append", nargs="+", default=[],
                        metavar=("SOURCE", "TARGET"),
                        help="Source and target guide paths, optionally followed by this guide's weight "
                             "(default 1/num_guides). Repeatable.")

    parser.add_argument("-output", type=str, default="output.png", dest="output_file", help="Output filename")
    parser.add_argument("-uniformity", type=float, default=3500.0, dest="uniformity_weight", help="Spatial uniformity scaling multiplier")
    parser.add_argument("-patchsize", type=int, default=5, dest="patch_size", help="Patch Match patching radius size")
    parser.add_argument("-pyramidlevels", type=int, default=-1, dest="num_pyramid_levels", help="Image pyramid scaling depth levels")
    parser.add_argument("-searchvoteiters", type=int, default=6, dest="num_search_vote_iters", help="Search and vote execution passes per level")
    parser.add_argument("-patchmatchiters", type=int, default=4, dest="num_patch_match_iters", help="Core Patch Match randomized propagation iterations")
    parser.add_argument("-stopthreshold", type=int, default=5, dest="stop_threshold", help="Parsed for CLI compatibility only; unused by this engine")
    parser.add_argument("-extrapass3x3", action="store_true", dest="extra_pass_3x3", help="Apply a final 3x3-patch refinement pass")

    try:
        target_args = argv[1:] if argv is not None else sys.argv[1:]
        args = parser.parse_args(target_args)

        def parse_weight(token):
            try:
                weight = float(token)
            except ValueError:
                parser.error(f"'{token}' is not a valid weight!")
            if weight < 0.0:
                parser.error("weights must be non-negative!")
            return weight

        if len(args.style) not in (1, 2):
            parser.error("-style takes <path> [weight]!")
        style_file = args.style[0]
        style_weight = parse_weight(args.style[1]) if len(args.style) == 2 else -1.0

        guides = []
        for tokens in args.guides:
            if len(tokens) not in (2, 3):
                parser.error("-guide takes <source> <target> [weight]!")
            weight = parse_weight(tokens[2]) if len(tokens) == 3 else -1.0
            guides.append({"source": tokens[0], "target": tokens[1], "weight": weight})

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

        return {
            "style_file": style_file,
            "style_weight": style_weight,
            "output_file": args.output_file,
            "guides": guides,
            "uniformity_weight": args.uniformity_weight,
            "patch_size": args.patch_size,
            "num_pyramid_levels": args.num_pyramid_levels,
            "num_search_vote_iters": args.num_search_vote_iters,
            "num_patch_match_iters": args.num_patch_match_iters,
            "stop_threshold": args.stop_threshold,
            "extra_pass_3x3": 1 if args.extra_pass_3x3 else 0,
        }

    except SystemExit:
        sys.exit(1)


# Sandbox check (run from the repo root: python arguments/parser.py)
if __name__ == "__main__":
    mock_cli = [
        "stylize.py",
        "-style", "painting.png", "2.0",
        "-guide", "src_flow.png", "tgt_flow.png", "10.5",
        "-guide", "src_seg.png", "tgt_seg.png",
        "-patchsize", "7",
        "-extrapass3x3"
    ]
    res = parse_arguments(mock_cli)
    import pprint
    print("Argument parsing test passed, parsed config:")
    pprint.pprint(res)

    assert res["style_file"] == "painting.png"
    assert res["style_weight"] == 2.0
    assert res["guides"][0] == {"source": "src_flow.png", "target": "tgt_flow.png", "weight": 10.5}
    assert res["guides"][1] == {"source": "src_seg.png", "target": "tgt_seg.png", "weight": -1.0}
    assert res["patch_size"] == 7
    assert res["extra_pass_3x3"] == 1

    # -style with no weight, -guide with no weight: both fall back to the -1.0 sentinel
    res_no_weight = parse_arguments(["stylize.py", "-style", "painting.png", "-guide", "a.png", "b.png"])
    assert res_no_weight["style_weight"] == -1.0
    assert res_no_weight["guides"][0]["weight"] == -1.0

    print("Weight parsing sanity checks passed ✓")
