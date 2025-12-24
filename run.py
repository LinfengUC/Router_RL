#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from router.config import REWARD_PROFILES, PRESETS
from router.train import run_experiment


def _add_train_args(p: argparse.ArgumentParser):
    p.add_argument(
        "--mode",
        choices=["base", "irt", "hard_gate"],
        default="base",
        help="Which router variant to run.",
    )
    p.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default="smoke",
        help="Training preset. Use 'full' to reproduce the original long runs.",
    )
    p.add_argument(
        "--profiles",
        default="accuracy_first",
        help="Comma-separated reward profiles, or 'all'.",
    )
    p.add_argument("--data-dir", default="data", help="Path to normalized data directory.")
    p.add_argument("--output-root", default="outputs", help="Where to write logs/models/metrics.")
    p.add_argument("--base-seed", type=int, default=42, help="Base seed; run i uses base_seed+i.")

    # Overrides (optional)
    p.add_argument("--runs", type=int, default=None, help="Override number of runs (seeds) per profile.")
    p.add_argument("--timesteps", type=int, default=None, help="Override total PPO timesteps.")
    p.add_argument("--eval-freq", type=int, default=None, help="Override evaluation frequency (train steps).")
    p.add_argument("--n-eval-episodes", type=int, default=None, help="Override #episodes for each eval.")

    # Device knobs
    p.add_argument(
        "--ppo-device",
        default="cpu",
        help="SB3 device for PPO. Default cpu (GPU often warns).",
    )
    p.add_argument(
        "--embedder-device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for the text embedder. 'auto' uses CUDA if available.",
    )


def _add_predict_args(p: argparse.ArgumentParser):
    p.add_argument("--checkpoint", required=True, help="Path to latest_model(.zip) or its run directory.")
    p.add_argument(
        "--run-config",
        default=None,
        help="Path to run_config.json (written during training). If omitted, we try to infer it.",
    )
    p.add_argument(
        "--input",
        required=True,
        help="CSV with at least columns: prompt_id, prompt. Extra columns are ignored.",
    )
    p.add_argument(
        "--output",
        default="outputs/predictions.csv",
        help="Where to write the routing decisions.",
    )

    p.add_argument(
        "--mode",
        choices=["auto", "base", "irt", "hard_gate"],
        default="auto",
        help="Routing mode. 'auto' reads it from run_config.json.",
    )

    # IRT options
    p.add_argument(
        "--irt-file",
        default=None,
        help="Optional IRT CSV. Supports missing one of {s_bert_pred,m_bert_pred} (mirrors the other).",
    )
    p.add_argument(
        "--irt-source",
        choices=["none", "csv", "dummy"],
        default="csv",
        help="Where IRT predictions come from for irt/hard_gate modes.",
    )

    # Model calling
    p.add_argument(
        "--use-stored-responses",
        action="store_true",
        help="Demo mode: call 'models' by reading data/responses/<split>/<model>.csv.",
    )
    p.add_argument(
        "--data-dir",
        default="data",
        help="Normalized data dir (needed for --use-stored-responses).",
    )
    p.add_argument(
        "--split",
        choices=["train", "test"],
        default="test",
        help="Which stored response split to use (demo only).",
    )

    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy (recommended for reproducible routing).",
    )
    p.add_argument(
        "--embedder-device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for the text embedder.",
    )


def _parse_profiles(arg: str) -> List[str]:
    if arg.strip().lower() == "all":
        return list(REWARD_PROFILES.keys())
    profiles = [x.strip() for x in arg.split(",") if x.strip()]
    if not profiles:
        raise SystemExit("No profiles specified. Use --profiles accuracy_first or --profiles all")
    return profiles


def _cmd_train(args: argparse.Namespace):
    profiles = _parse_profiles(args.profiles)
    run_experiment(
        mode=args.mode,
        preset_name=args.preset,
        profiles=profiles,
        data_dir=args.data_dir,
        output_root=args.output_root,
        base_seed=args.base_seed,
        runs_override=args.runs,
        timesteps_override=args.timesteps,
        eval_freq_override=args.eval_freq,
        n_eval_episodes_override=args.n_eval_episodes,
        embedder_device=args.embedder_device,
        ppo_device=args.ppo_device,
    )


def _cmd_predict(args: argparse.Namespace):
    from router.infer import (
        DummySingleIRTProvider,
        StoredResponseCaller,
        load_irt_provider_from_path,
        load_prompt_list_csv,
        load_router,
        route_prompt,
        write_route_results_csv,
    )
    from router.embedder import TextEmbedder

    sb3_model, cfg = load_router(args.checkpoint, run_config_path=args.run_config)

    mode = cfg.get("mode") if args.mode == "auto" else args.mode
    if mode not in {"base", "irt", "hard_gate"}:
        raise SystemExit(f"Invalid mode {mode!r}. Use --mode base|irt|hard_gate (or auto).")

    model_names = list(cfg.get("common_models", []))
    if not model_names:
        raise SystemExit("run_config.json missing 'common_models'.")

    base_model_name = cfg.get("base_model_name", "deepseek_V3.2_no_reasoning")
    embedder_model_name = cfg.get("embedder_model_name", "sentence-transformers/all-mpnet-base-v2")

    embedder = TextEmbedder(model_name=embedder_model_name, device=args.embedder_device)

    # IRT provider
    irt_provider = None
    if mode in ("irt", "hard_gate"):
        if args.irt_source == "none":
            irt_provider = None
        elif args.irt_source == "dummy":
            irt_provider = DummySingleIRTProvider()
        else:
            if not args.irt_file:
                # In demo workflows, auto-pick data/irt/<split>.csv if it exists.
                cand = os.path.join(args.data_dir, "irt", f"{args.split}.csv")
                args.irt_file = cand if os.path.exists(cand) else None
            irt_provider = load_irt_provider_from_path(args.irt_file)

    # Model caller
    if not args.use_stored_responses:
        raise SystemExit(
            "For now, predict mode requires --use-stored-responses (demo).\n"
            "To use real model endpoints, implement a ModelCaller and call router.infer.route_prompt() from your code."
        )

    responses_dir = os.path.join(args.data_dir, "responses")
    model_caller = StoredResponseCaller(responses_dir, args.split, model_names)

    prompts = load_prompt_list_csv(args.input)
    results = []
    for pid, prompt in prompts:
        rr = route_prompt(
            sb3_model=sb3_model,
            mode=mode,
            embedder=embedder,
            model_names=model_names,
            model_caller=model_caller,
            prompt_id=pid,
            prompt=prompt,
            irt_provider=irt_provider,
            base_model_name=base_model_name,
            deterministic=bool(args.deterministic),
        )
        results.append(rr)

    write_route_results_csv(args.output, results, model_names=model_names)
    print(f"Wrote {len(results)} predictions to: {args.output}")


def main(argv: Optional[List[str]] = None):
    argv = list(sys.argv[1:] if argv is None else argv)

    # Backward-compatible default: if no explicit subcommand, assume "train".
    if len(argv) == 0 or argv[0] not in {"train", "predict"}:
        argv = ["train"] + argv

    parser = argparse.ArgumentParser(
        description="Modular RL router runner (train / predict).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train router policies.")
    _add_train_args(p_train)
    p_train.set_defaults(func=_cmd_train)

    p_pred = sub.add_parser("predict", help="Use a trained router to route prompts.")
    _add_predict_args(p_pred)
    p_pred.set_defaults(func=_cmd_predict)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
