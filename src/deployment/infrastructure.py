"""AWS infrastructure setup and verification for SageMaker deployment."""

from __future__ import annotations

import json
from typing import Any

import boto3
from botocore.exceptions import ClientError

from src.config.settings import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

# Cost per hour for common SageMaker GPU instances (approximate, us-east-1)
_INSTANCE_COSTS: dict[str, float] = {
    "ml.g4dn.xlarge": 0.736,
    "ml.g4dn.2xlarge": 1.12,
    "ml.g5.xlarge": 1.408,
    "ml.g5.2xlarge": 1.515,
    "ml.g5.4xlarge": 2.03,
    "ml.g5.12xlarge": 7.09,
    "ml.p3.2xlarge": 3.825,
    "ml.p4d.24xlarge": 37.688,
    "ml.inf2.xlarge": 0.758,
}


class AWSInfrastructure:
    """Manage AWS infrastructure for SageMaker deployments.

    Args:
        region: AWS region.
        boto_session: Optional boto3 session (for testing).
    """

    def __init__(
        self,
        region: str | None = None,
        boto_session: Any | None = None,
    ) -> None:
        self.region = region or settings.aws_region
        session = boto_session or boto3.Session(region_name=self.region)
        self.s3_client = session.client("s3")
        self.iam_client = session.client("iam")
        self.sts_client = session.client("sts")

    def setup_s3_bucket(self, bucket_name: str) -> bool:
        """Create an S3 bucket if it doesn't exist.

        Args:
            bucket_name: S3 bucket name.

        Returns:
            True if created, False if already exists.
        """
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            log.info("s3_bucket_exists", bucket=bucket_name)
            return False
        except ClientError:
            pass

        create_kwargs: dict[str, Any] = {"Bucket": bucket_name}
        if self.region != "us-east-1":
            create_kwargs["CreateBucketConfiguration"] = {
                "LocationConstraint": self.region,
            }

        self.s3_client.create_bucket(**create_kwargs)
        log.info("s3_bucket_created", bucket=bucket_name, region=self.region)
        return True

    def setup_iam_role(self, role_name: str = "SageMakerExecutionRole") -> str:
        """Create a SageMaker execution role if it doesn't exist.

        Args:
            role_name: IAM role name.

        Returns:
            Role ARN.
        """
        try:
            response = self.iam_client.get_role(RoleName=role_name)
            arn = response["Role"]["Arn"]
            log.info("iam_role_exists", role=role_name, arn=arn)
            return arn
        except ClientError:
            pass

        assume_role_policy = json.dumps({
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "sagemaker.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        })

        response = self.iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=assume_role_policy,
            Description="SageMaker execution role for LLM fine-tuning pipeline",
        )
        arn = response["Role"]["Arn"]

        # Attach required policies
        policies = [
            "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
            "arn:aws:iam::aws:policy/AmazonS3FullAccess",
        ]
        for policy_arn in policies:
            self.iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)

        log.info("iam_role_created", role=role_name, arn=arn)
        return arn

    def verify_permissions(self) -> dict[str, bool]:
        """Verify required AWS permissions.

        Returns:
            Dict mapping permission checks to pass/fail status.
        """
        results: dict[str, bool] = {}

        # Check STS identity
        try:
            identity = self.sts_client.get_caller_identity()
            results["aws_identity"] = True
            log.info("aws_identity_verified", account=identity["Account"])
        except ClientError:
            results["aws_identity"] = False

        # Check S3 access
        try:
            if settings.s3_bucket:
                self.s3_client.head_bucket(Bucket=settings.s3_bucket)
                results["s3_access"] = True
            else:
                results["s3_access"] = False
        except ClientError:
            results["s3_access"] = False

        # Check IAM role exists
        try:
            if settings.aws_sagemaker_role:
                role_name = settings.aws_sagemaker_role.split("/")[-1]
                self.iam_client.get_role(RoleName=role_name)
                results["iam_role"] = True
            else:
                results["iam_role"] = False
        except ClientError:
            results["iam_role"] = False

        log.info("permissions_verified", results=results)
        return results

    @staticmethod
    def estimate_cost(instance_type: str, hours: float = 1.0) -> dict[str, Any]:
        """Estimate deployment cost for a given instance type.

        Args:
            instance_type: SageMaker instance type.
            hours: Number of hours to estimate.

        Returns:
            Dict with hourly_cost, total_cost, instance_type.
        """
        hourly = _INSTANCE_COSTS.get(instance_type, 0.0)
        return {
            "instance_type": instance_type,
            "hourly_cost_usd": hourly,
            "total_cost_usd": round(hourly * hours, 2),
            "hours": hours,
            "known_pricing": instance_type in _INSTANCE_COSTS,
        }

    @staticmethod
    def get_available_instances() -> list[str]:
        """List available GPU instance types for SageMaker inference.

        Returns:
            List of instance type strings.
        """
        return sorted(_INSTANCE_COSTS.keys())
