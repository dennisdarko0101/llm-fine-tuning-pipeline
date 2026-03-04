"""Tests for AWS infrastructure setup (mocked AWS — no real calls)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from src.deployment.infrastructure import AWSInfrastructure


@pytest.fixture()
def mock_boto_session() -> MagicMock:
    """Create a mock boto3 session."""
    session = MagicMock()
    s3 = MagicMock()
    iam = MagicMock()
    sts = MagicMock()
    sts.get_caller_identity.return_value = {
        "Account": "123456789012",
        "Arn": "arn:aws:iam::123456789012:user/test",
    }

    def client_factory(service_name: str, **kwargs) -> MagicMock:
        return {"s3": s3, "iam": iam, "sts": sts}[service_name]

    session.client = MagicMock(side_effect=client_factory)
    return session


@pytest.fixture()
def infra(mock_boto_session: MagicMock) -> AWSInfrastructure:
    """Create infrastructure manager with mocked AWS."""
    return AWSInfrastructure(region="us-east-1", boto_session=mock_boto_session)


class TestSetupS3Bucket:
    """Tests for S3 bucket setup."""

    def test_creates_bucket_when_not_exists(self, infra: AWSInfrastructure) -> None:
        infra.s3_client.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadBucket"
        )
        result = infra.setup_s3_bucket("my-bucket")
        assert result is True
        infra.s3_client.create_bucket.assert_called_once()

    def test_skips_if_bucket_exists(self, infra: AWSInfrastructure) -> None:
        infra.s3_client.head_bucket.return_value = {}
        result = infra.setup_s3_bucket("my-bucket")
        assert result is False
        infra.s3_client.create_bucket.assert_not_called()

    def test_creates_with_location_constraint(self) -> None:
        session = MagicMock()
        s3 = MagicMock()
        s3.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "404"}}, "HeadBucket"
        )
        session.client = MagicMock(return_value=s3)
        infra = AWSInfrastructure(region="eu-west-1", boto_session=session)
        infra.setup_s3_bucket("my-bucket")
        call_kwargs = s3.create_bucket.call_args[1]
        assert "CreateBucketConfiguration" in call_kwargs


class TestSetupIAMRole:
    """Tests for IAM role setup."""

    def test_returns_existing_role(self, infra: AWSInfrastructure) -> None:
        infra.iam_client.get_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123:role/SageMakerExecutionRole"}
        }
        arn = infra.setup_iam_role()
        assert "SageMakerExecutionRole" in arn
        infra.iam_client.create_role.assert_not_called()

    def test_creates_role_when_not_exists(self, infra: AWSInfrastructure) -> None:
        infra.iam_client.get_role.side_effect = ClientError(
            {"Error": {"Code": "NoSuchEntity"}}, "GetRole"
        )
        infra.iam_client.create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123:role/SageMakerExecutionRole"}
        }
        arn = infra.setup_iam_role()
        assert "SageMakerExecutionRole" in arn
        infra.iam_client.create_role.assert_called_once()
        # Should attach 2 policies
        assert infra.iam_client.attach_role_policy.call_count == 2


class TestVerifyPermissions:
    """Tests for permission verification."""

    def test_all_pass(self, infra: AWSInfrastructure) -> None:
        with patch("src.deployment.infrastructure.settings") as mock_settings:
            mock_settings.s3_bucket = "my-bucket"
            mock_settings.aws_sagemaker_role = "arn:aws:iam::123:role/SageMakerRole"
            infra.s3_client.head_bucket.return_value = {}
            infra.iam_client.get_role.return_value = {"Role": {"Arn": "arn"}}
            results = infra.verify_permissions()
        assert results["aws_identity"] is True
        assert results["s3_access"] is True
        assert results["iam_role"] is True

    def test_identity_failure(self, infra: AWSInfrastructure) -> None:
        infra.sts_client.get_caller_identity.side_effect = ClientError(
            {"Error": {"Code": "403"}}, "GetCallerIdentity"
        )
        with patch("src.deployment.infrastructure.settings") as mock_settings:
            mock_settings.s3_bucket = ""
            mock_settings.aws_sagemaker_role = ""
            results = infra.verify_permissions()
        assert results["aws_identity"] is False

    def test_s3_no_bucket_configured(self, infra: AWSInfrastructure) -> None:
        with patch("src.deployment.infrastructure.settings") as mock_settings:
            mock_settings.s3_bucket = ""
            mock_settings.aws_sagemaker_role = ""
            results = infra.verify_permissions()
        assert results["s3_access"] is False

    def test_s3_access_denied(self, infra: AWSInfrastructure) -> None:
        infra.s3_client.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "403"}}, "HeadBucket"
        )
        with patch("src.deployment.infrastructure.settings") as mock_settings:
            mock_settings.s3_bucket = "my-bucket"
            mock_settings.aws_sagemaker_role = ""
            results = infra.verify_permissions()
        assert results["s3_access"] is False


class TestCostEstimation:
    """Tests for cost estimation."""

    def test_known_instance(self) -> None:
        cost = AWSInfrastructure.estimate_cost("ml.g5.xlarge", hours=10)
        assert cost["instance_type"] == "ml.g5.xlarge"
        assert cost["hourly_cost_usd"] > 0
        assert cost["total_cost_usd"] == pytest.approx(cost["hourly_cost_usd"] * 10, abs=0.01)
        assert cost["known_pricing"] is True

    def test_unknown_instance(self) -> None:
        cost = AWSInfrastructure.estimate_cost("ml.unknown.xlarge")
        assert cost["hourly_cost_usd"] == 0.0
        assert cost["known_pricing"] is False

    def test_default_hours(self) -> None:
        cost = AWSInfrastructure.estimate_cost("ml.g5.xlarge")
        assert cost["hours"] == 1.0


class TestAvailableInstances:
    """Tests for instance listing."""

    def test_returns_list(self) -> None:
        instances = AWSInfrastructure.get_available_instances()
        assert isinstance(instances, list)
        assert len(instances) > 0
        assert "ml.g5.xlarge" in instances

    def test_instances_sorted(self) -> None:
        instances = AWSInfrastructure.get_available_instances()
        assert instances == sorted(instances)
