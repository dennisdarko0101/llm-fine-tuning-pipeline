"""AWS SageMaker deployment for fine-tuned models."""

from __future__ import annotations

import tarfile
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

from src.config.settings import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

# HuggingFace Deep Learning Container image URI pattern
_HF_INFERENCE_IMAGE = (
    "763104351884.dkr.ecr.{region}.amazonaws.com/"
    "huggingface-pytorch-inference:{pytorch_version}-transformers{transformers_version}-"
    "gpu-py{python_version}-cu{cuda_version}-ubuntu{ubuntu_version}"
)

_DEFAULT_IMAGE_CONFIG = {
    "pytorch_version": "2.1.0",
    "transformers_version": "4.36.0",
    "python_version": "310",
    "cuda_version": "121",
    "ubuntu_version": "20.04",
}


@dataclass
class EndpointInfo:
    """Information about a deployed SageMaker endpoint."""

    name: str = ""
    arn: str = ""
    instance_type: str = ""
    status: str = ""
    creation_time: str = ""
    s3_model_uri: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        from dataclasses import asdict
        return asdict(self)


class SageMakerDeployer:
    """Deploy fine-tuned models to AWS SageMaker.

    Args:
        region: AWS region.
        role_arn: SageMaker execution role ARN.
        s3_bucket: S3 bucket for model artifacts.
        boto_session: Optional boto3 session (for testing).
    """

    def __init__(
        self,
        region: str | None = None,
        role_arn: str | None = None,
        s3_bucket: str | None = None,
        boto_session: Any | None = None,
    ) -> None:
        self.region = region or settings.aws_region
        self.role_arn = role_arn or settings.aws_sagemaker_role
        self.s3_bucket = s3_bucket or settings.s3_bucket
        session = boto_session or boto3.Session(region_name=self.region)
        self.s3_client = session.client("s3")
        self.sm_client = session.client("sagemaker")

    def package_model(self, model_path: str | Path, s3_prefix: str = "models") -> str:
        """Package model files into tar.gz and upload to S3.

        Args:
            model_path: Local path to model directory.
            s3_prefix: S3 key prefix.

        Returns:
            S3 URI of the uploaded model artifact.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        s3_key = f"{s3_prefix}/{model_path.name}-{timestamp}/model.tar.gz"

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = tmp.name

        with tarfile.open(tmp_path, "w:gz") as tar:
            for file in model_path.iterdir():
                tar.add(str(file), arcname=file.name)

        self.s3_client.upload_file(tmp_path, self.s3_bucket, s3_key)
        Path(tmp_path).unlink(missing_ok=True)

        s3_uri = f"s3://{self.s3_bucket}/{s3_key}"
        log.info("model_packaged", s3_uri=s3_uri)
        return s3_uri

    def create_model(
        self,
        s3_model_uri: str,
        model_name: str | None = None,
        instance_type: str = "ml.g5.xlarge",
        image_uri: str | None = None,
    ) -> str:
        """Create a SageMaker model from an S3 artifact.

        Args:
            s3_model_uri: S3 URI of model.tar.gz.
            model_name: SageMaker model name (auto-generated if None).
            instance_type: Target instance type.
            image_uri: Custom container image URI.

        Returns:
            SageMaker model name.
        """
        if not model_name:
            timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            model_name = f"llm-ft-{timestamp}"

        if not image_uri:
            image_uri = _HF_INFERENCE_IMAGE.format(
                region=self.region, **_DEFAULT_IMAGE_CONFIG
            )

        self.sm_client.create_model(
            ModelName=model_name,
            ExecutionRoleArn=self.role_arn,
            PrimaryContainer={
                "Image": image_uri,
                "ModelDataUrl": s3_model_uri,
                "Environment": {
                    "HF_TASK": "text-generation",
                    "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                },
            },
        )

        log.info("sagemaker_model_created", model_name=model_name, s3_uri=s3_model_uri)
        return model_name

    def deploy_endpoint(
        self,
        model_name: str,
        instance_type: str = "ml.g5.xlarge",
        initial_instance_count: int = 1,
        endpoint_name: str | None = None,
        wait: bool = True,
    ) -> EndpointInfo:
        """Deploy a model to a SageMaker endpoint.

        Args:
            model_name: SageMaker model name.
            instance_type: EC2 instance type.
            initial_instance_count: Number of instances.
            endpoint_name: Endpoint name (defaults to model_name).
            wait: Wait for endpoint to be InService.

        Returns:
            EndpointInfo with deployment details.
        """
        endpoint_name = endpoint_name or model_name
        config_name = f"{endpoint_name}-config"

        # Create endpoint config
        self.sm_client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    "VariantName": "primary",
                    "ModelName": model_name,
                    "InstanceType": instance_type,
                    "InitialInstanceCount": initial_instance_count,
                },
            ],
        )

        # Create or update endpoint
        try:
            self.sm_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name,
            )
            log.info("endpoint_creating", endpoint=endpoint_name)
        except ClientError as e:
            if "Cannot create already existing" in str(e):
                self.sm_client.update_endpoint(
                    EndpointName=endpoint_name,
                    EndpointConfigName=config_name,
                )
                log.info("endpoint_updating", endpoint=endpoint_name)
            else:
                raise

        if wait:
            self._wait_for_endpoint(endpoint_name)

        status = self.get_endpoint_status(endpoint_name)
        return EndpointInfo(
            name=endpoint_name,
            arn=status.get("EndpointArn", ""),
            instance_type=instance_type,
            status=status.get("EndpointStatus", "Unknown"),
            creation_time=str(status.get("CreationTime", "")),
            s3_model_uri="",
        )

    def invoke_endpoint(self, endpoint_name: str, payload: str) -> str:
        """Send an inference request to a deployed endpoint.

        Args:
            endpoint_name: SageMaker endpoint name.
            payload: JSON string payload.

        Returns:
            Response body as string.
        """
        runtime = boto3.client("sagemaker-runtime", region_name=self.region)
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=payload.encode("utf-8"),
        )
        result = response["Body"].read().decode("utf-8")
        log.info("endpoint_invoked", endpoint=endpoint_name)
        return result

    def delete_endpoint(self, endpoint_name: str) -> None:
        """Delete a SageMaker endpoint, config, and model.

        Args:
            endpoint_name: Endpoint name to delete.
        """
        # Delete endpoint
        try:
            self.sm_client.delete_endpoint(EndpointName=endpoint_name)
            log.info("endpoint_deleted", endpoint=endpoint_name)
        except ClientError as e:
            log.warning("endpoint_delete_failed", endpoint=endpoint_name, error=str(e))

        # Delete endpoint config
        config_name = f"{endpoint_name}-config"
        try:
            self.sm_client.delete_endpoint_config(EndpointConfigName=config_name)
            log.info("endpoint_config_deleted", config=config_name)
        except ClientError:
            pass

        # Delete model
        try:
            self.sm_client.delete_model(ModelName=endpoint_name)
            log.info("sagemaker_model_deleted", model=endpoint_name)
        except ClientError:
            pass

    def get_endpoint_status(self, endpoint_name: str) -> dict[str, Any]:
        """Get the status of a SageMaker endpoint.

        Args:
            endpoint_name: Endpoint name.

        Returns:
            Dict with endpoint status information.
        """
        try:
            response = self.sm_client.describe_endpoint(EndpointName=endpoint_name)
            return {
                "EndpointName": response["EndpointName"],
                "EndpointArn": response["EndpointArn"],
                "EndpointStatus": response["EndpointStatus"],
                "CreationTime": str(response.get("CreationTime", "")),
            }
        except ClientError as e:
            log.warning("endpoint_status_failed", endpoint=endpoint_name, error=str(e))
            return {"EndpointName": endpoint_name, "EndpointStatus": "NotFound"}

    def _wait_for_endpoint(
        self, endpoint_name: str, timeout: int = 900, poll_interval: int = 30
    ) -> None:
        """Wait for endpoint to reach InService status."""
        start = time.time()
        while time.time() - start < timeout:
            status = self.get_endpoint_status(endpoint_name)
            current = status.get("EndpointStatus", "")
            if current == "InService":
                log.info("endpoint_in_service", endpoint=endpoint_name)
                return
            if current == "Failed":
                raise RuntimeError(f"Endpoint {endpoint_name} failed to deploy")
            log.info("endpoint_waiting", endpoint=endpoint_name, status=current)
            time.sleep(poll_interval)
        raise TimeoutError(f"Endpoint {endpoint_name} did not reach InService within {timeout}s")
