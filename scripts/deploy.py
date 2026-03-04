"""CLI script for SageMaker model deployment.

Usage:
    python scripts/deploy.py --action package --model-path outputs/final
    python scripts/deploy.py --action deploy --model-path outputs/final --endpoint-name my-llm
    python scripts/deploy.py --action invoke --endpoint-name my-llm --prompt "Hello"
    python scripts/deploy.py --action status --endpoint-name my-llm
    python scripts/deploy.py --action delete --endpoint-name my-llm
"""

from __future__ import annotations

import argparse
import json
import sys

from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Run deployment actions."""
    parser = argparse.ArgumentParser(description="Deploy models to AWS SageMaker")
    parser.add_argument(
        "--action",
        required=True,
        choices=["package", "deploy", "invoke", "delete", "status", "estimate"],
        help="Deployment action to perform",
    )
    parser.add_argument("--model-path", default=None, help="Path to model directory")
    parser.add_argument("--endpoint-name", default=None, help="SageMaker endpoint name")
    parser.add_argument(
        "--instance-type", default="ml.g5.xlarge", help="SageMaker instance type"
    )
    parser.add_argument("--prompt", default=None, help="Prompt for invoke action")
    parser.add_argument("--hours", type=float, default=24.0, help="Hours for cost estimate")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompts")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)

    if args.action == "estimate":
        _action_estimate(args)
    elif args.action == "package":
        _action_package(args)
    elif args.action == "deploy":
        _action_deploy(args)
    elif args.action == "invoke":
        _action_invoke(args)
    elif args.action == "status":
        _action_status(args)
    elif args.action == "delete":
        _action_delete(args)


def _action_estimate(args: argparse.Namespace) -> None:
    """Print cost estimate for an instance type."""
    from src.deployment.infrastructure import AWSInfrastructure

    cost = AWSInfrastructure.estimate_cost(args.instance_type, hours=args.hours)
    print(f"\nCost Estimate for {cost['instance_type']}:")
    print(f"  Hourly: ${cost['hourly_cost_usd']:.3f}/hr")
    print(f"  Total ({cost['hours']}h): ${cost['total_cost_usd']:.2f}")
    if not cost["known_pricing"]:
        print("  (pricing not in database — verify on AWS)")


def _action_package(args: argparse.Namespace) -> None:
    """Package and upload model to S3."""
    if not args.model_path:
        log.error("missing_model_path")
        sys.exit(1)

    from src.deployment.sagemaker import SageMakerDeployer

    deployer = SageMakerDeployer()
    s3_uri = deployer.package_model(args.model_path)
    print(f"\nModel packaged and uploaded to: {s3_uri}")


def _action_deploy(args: argparse.Namespace) -> None:
    """Deploy model to SageMaker endpoint."""
    if not args.model_path:
        log.error("missing_model_path")
        sys.exit(1)

    from src.deployment.infrastructure import AWSInfrastructure
    from src.deployment.sagemaker import SageMakerDeployer

    # Show cost estimate
    cost = AWSInfrastructure.estimate_cost(args.instance_type, hours=args.hours)
    print(f"\nDeploying to {args.instance_type}:")
    print(f"  Estimated cost: ${cost['hourly_cost_usd']:.3f}/hr (${cost['total_cost_usd']:.2f} for {args.hours}h)")

    if not args.yes:
        confirm = input("\nProceed with deployment? [y/N] ").strip().lower()
        if confirm != "y":
            print("Deployment cancelled.")
            return

    deployer = SageMakerDeployer()

    # Package model
    print("\nPackaging model...")
    s3_uri = deployer.package_model(args.model_path)

    # Create model
    print("Creating SageMaker model...")
    endpoint_name = args.endpoint_name or "llm-ft-endpoint"
    model_name = deployer.create_model(s3_uri, model_name=endpoint_name)

    # Deploy endpoint
    print(f"Deploying endpoint '{endpoint_name}'...")
    info = deployer.deploy_endpoint(
        model_name, instance_type=args.instance_type, endpoint_name=endpoint_name
    )

    print("\nEndpoint deployed:")
    print(f"  Name: {info.name}")
    print(f"  Status: {info.status}")
    print(f"  Instance: {info.instance_type}")


def _action_invoke(args: argparse.Namespace) -> None:
    """Invoke a deployed endpoint."""
    if not args.endpoint_name:
        log.error("missing_endpoint_name")
        sys.exit(1)
    if not args.prompt:
        log.error("missing_prompt")
        sys.exit(1)

    from src.deployment.sagemaker import SageMakerDeployer

    deployer = SageMakerDeployer()
    payload = json.dumps({"inputs": args.prompt})
    response = deployer.invoke_endpoint(args.endpoint_name, payload)
    print(f"\nResponse:\n{response}")


def _action_status(args: argparse.Namespace) -> None:
    """Check endpoint status."""
    if not args.endpoint_name:
        log.error("missing_endpoint_name")
        sys.exit(1)

    from src.deployment.sagemaker import SageMakerDeployer

    deployer = SageMakerDeployer()
    status = deployer.get_endpoint_status(args.endpoint_name)
    print("\nEndpoint Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")


def _action_delete(args: argparse.Namespace) -> None:
    """Delete an endpoint."""
    if not args.endpoint_name:
        log.error("missing_endpoint_name")
        sys.exit(1)

    if not args.yes:
        confirm = input(f"Delete endpoint '{args.endpoint_name}'? [y/N] ").strip().lower()
        if confirm != "y":
            print("Deletion cancelled.")
            return

    from src.deployment.sagemaker import SageMakerDeployer

    deployer = SageMakerDeployer()
    deployer.delete_endpoint(args.endpoint_name)
    print(f"\nEndpoint '{args.endpoint_name}' deleted.")


if __name__ == "__main__":
    main()
