"""Tests for SageMaker deployment (mocked AWS — no real calls)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.deployment.sagemaker import EndpointInfo, SageMakerDeployer


@pytest.fixture()
def mock_boto_session() -> MagicMock:
    """Create a mock boto3 session."""
    session = MagicMock()
    session.client = MagicMock(side_effect=_mock_client_factory)
    return session


def _mock_client_factory(service_name: str, **kwargs) -> MagicMock:
    """Create mock clients per service."""
    client = MagicMock()
    if service_name == "sagemaker":
        client.describe_endpoint.return_value = {
            "EndpointName": "test-endpoint",
            "EndpointArn": "arn:aws:sagemaker:us-east-1:123456789:endpoint/test-endpoint",
            "EndpointStatus": "InService",
            "CreationTime": "2024-01-01T00:00:00Z",
        }
    return client


@pytest.fixture()
def deployer(mock_boto_session: MagicMock) -> SageMakerDeployer:
    """Create a deployer with mocked AWS."""
    with patch("src.deployment.sagemaker.settings") as mock_settings:
        mock_settings.aws_region = "us-east-1"
        mock_settings.aws_sagemaker_role = "arn:aws:iam::123456789:role/SageMakerRole"
        mock_settings.s3_bucket = "test-bucket"
        return SageMakerDeployer(
            region="us-east-1",
            role_arn="arn:aws:iam::123456789:role/SageMakerRole",
            s3_bucket="test-bucket",
            boto_session=mock_boto_session,
        )


class TestEndpointInfo:
    """Tests for EndpointInfo dataclass."""

    def test_default_values(self) -> None:
        info = EndpointInfo()
        assert info.name == ""
        assert info.status == ""

    def test_to_dict(self) -> None:
        info = EndpointInfo(name="ep", status="InService", instance_type="ml.g5.xlarge")
        d = info.to_dict()
        assert d["name"] == "ep"
        assert d["instance_type"] == "ml.g5.xlarge"


class TestPackageModel:
    """Tests for model packaging."""

    def test_package_model(self, deployer: SageMakerDeployer, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"model": "test"}')
        (model_dir / "model.safetensors").write_bytes(b"fake weights")

        s3_uri = deployer.package_model(model_dir)
        assert s3_uri.startswith("s3://test-bucket/models/")
        assert s3_uri.endswith("model.tar.gz")
        deployer.s3_client.upload_file.assert_called_once()

    def test_package_model_not_found(self, deployer: SageMakerDeployer) -> None:
        with pytest.raises(FileNotFoundError):
            deployer.package_model("/nonexistent/path")


class TestCreateModel:
    """Tests for SageMaker model creation."""

    def test_create_model(self, deployer: SageMakerDeployer) -> None:
        name = deployer.create_model(
            "s3://bucket/model.tar.gz", model_name="my-model"
        )
        assert name == "my-model"
        deployer.sm_client.create_model.assert_called_once()
        call_kwargs = deployer.sm_client.create_model.call_args[1]
        assert call_kwargs["ModelName"] == "my-model"
        assert call_kwargs["ExecutionRoleArn"] == deployer.role_arn

    def test_create_model_auto_name(self, deployer: SageMakerDeployer) -> None:
        name = deployer.create_model("s3://bucket/model.tar.gz")
        assert name.startswith("llm-ft-")


class TestDeployEndpoint:
    """Tests for endpoint deployment."""

    def test_deploy_creates_config_and_endpoint(self, deployer: SageMakerDeployer) -> None:
        info = deployer.deploy_endpoint(
            "my-model",
            instance_type="ml.g5.xlarge",
            endpoint_name="my-ep",
            wait=False,
        )
        deployer.sm_client.create_endpoint_config.assert_called_once()
        deployer.sm_client.create_endpoint.assert_called_once()
        assert info.name == "my-ep"
        assert info.instance_type == "ml.g5.xlarge"

    def test_deploy_endpoint_config_params(self, deployer: SageMakerDeployer) -> None:
        deployer.deploy_endpoint("model", endpoint_name="ep", wait=False)
        config_call = deployer.sm_client.create_endpoint_config.call_args[1]
        variant = config_call["ProductionVariants"][0]
        assert variant["ModelName"] == "model"
        assert variant["InitialInstanceCount"] == 1


class TestInvokeEndpoint:
    """Tests for endpoint invocation."""

    def test_invoke_endpoint(self, deployer: SageMakerDeployer) -> None:
        with patch("src.deployment.sagemaker.boto3") as mock_boto:
            runtime = MagicMock()
            runtime.invoke_endpoint.return_value = {
                "Body": MagicMock(read=MagicMock(return_value=b'{"generated_text": "Hello"}'))
            }
            mock_boto.client.return_value = runtime

            result = deployer.invoke_endpoint("my-ep", '{"inputs": "Hi"}')
            assert "Hello" in result


class TestDeleteEndpoint:
    """Tests for endpoint deletion."""

    def test_delete_endpoint(self, deployer: SageMakerDeployer) -> None:
        deployer.delete_endpoint("my-ep")
        deployer.sm_client.delete_endpoint.assert_called_once_with(EndpointName="my-ep")
        deployer.sm_client.delete_endpoint_config.assert_called_once()
        deployer.sm_client.delete_model.assert_called_once()


class TestGetEndpointStatus:
    """Tests for endpoint status checking."""

    def test_get_status(self, deployer: SageMakerDeployer) -> None:
        status = deployer.get_endpoint_status("test-endpoint")
        assert status["EndpointStatus"] == "InService"
        assert "EndpointArn" in status

    def test_get_status_not_found(self, deployer: SageMakerDeployer) -> None:
        from botocore.exceptions import ClientError

        deployer.sm_client.describe_endpoint.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not found"}}, "DescribeEndpoint"
        )
        status = deployer.get_endpoint_status("nonexistent")
        assert status["EndpointStatus"] == "NotFound"
