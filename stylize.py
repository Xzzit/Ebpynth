import sys

from arguments import parse_arguments
from utils import load_image_to_vram, save_image_from_vram, merge_guides, plan_pyramid
from synthesis import run_pyramid


def main():
    # 🐍 Task A: Pass live system CLI arguments straight into the custom parsing token scanner
    config = parse_arguments(sys.argv)

    # 🐍 Task B: Style painting goes straight into VRAM as a (H, W, C) uint8 contiguous tensor
    source_style = load_image_to_vram(config["style_file"])
    style_h, style_w, style_c = source_style.shape

    # 🐍 Task C: Twist every guide pair's channels into two mega feature tensors
    source_guides, target_guides, guide_channels = merge_guides(
        config["guides"], source_style.shape, config["style_file"])
    target_h, target_w = target_guides.shape[0], target_guides.shape[1]

    # 🐍 Task D: Pure scalar math — pyramid schedule and per-channel weight vectors (stays on CPU)
    plan = plan_pyramid(config, source_style.shape, target_guides.shape, guide_channels)

    # Hyper-parameter roll call, matching the original CLI printout (ebsynth.cpp lines ~430-436)
    print(f"uniformity: {config['uniformity_weight']:.0f}")
    print(f"patchsize: {config['patch_size']}")
    print(f"pyramidlevels: {plan['num_pyramid_levels']}")
    print(f"searchvoteiters: {config['num_search_vote_iters']}")
    print(f"patchmatchiters: {config['num_patch_match_iters']}")
    print(f"stopthreshold: {config['stop_threshold']}")
    print(f"extrapass3x3: {'yes' if config['extra_pass_3x3'] else 'no'}")

    # ⚡ Phase 2: hand everything over to the synthesis engine (Tasks F-J). It's built as
    # pure-functional PyTorch (every stage returns a new tensor rather than writing into
    # one shared buffer), so output_image's role is just a shape/dtype sanity check now,
    # not a literal write-target the way the pre-CUDA-bridge plan originally assumed.
    _, output_image = run_pyramid(
        source_style, source_guides, target_guides,
        plan["style_weights"], plan["guide_weights"], config["patch_size"],
        plan["num_pyramid_levels"], plan["search_vote_iters_per_level"], plan["patch_match_iters_per_level"],
        uniformity_weight=config["uniformity_weight"],
        extra_pass_3x3=bool(config["extra_pass_3x3"]))
    # 🐍 Task E's contract, checked rather than pre-allocated: (H_target, W_target, C_style)
    assert output_image.shape == (target_h, target_w, style_c), "engine output shape drifted from the planned canvas"

    # 🐍 Task F: Land the finished canvas back onto the disk
    save_image_from_vram(output_image, config["output_file"])
    print(f"result was written to {config['output_file']}")


if __name__ == "__main__":
    main()
