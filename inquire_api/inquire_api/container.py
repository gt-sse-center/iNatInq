"""Container utilities."""

from dataclasses import dataclass, field

import docker
from loguru import logger


@dataclass
class ContainerConfig:
    """Container configuration."""

    name: str = "inatinq-qdrant"
    image: str = "qdrant/qdrant:latest"
    hostname = "localhost"
    ports: dict[int, int] = field(default_factory=lambda: {6333: 7333, 6334: 7334})
    environment: dict = field(default_factory=dict)
    volumes: list[str] = field(default_factory=list)
    command: str | list = ""
    security_opt: list[str] = field(default_factory=list)
    healthcheck: dict[str, int | str] = field(
        default_factory=lambda: {
            "test": "curl -s http://localhost:6333/healthz | grep -q 'healthz check passed' || exit 1",
            "interval": 3 * 10**9,
            "timeout": 2 * 10**9,
            "retries": 3,
        }
    )
    network: str = ""


class Container:
    """Class to manage a docker container."""

    def __init__(
        self,
        *,
        container_cfg: ContainerConfig | None = None,
        remove_on_stop: bool = False,
        auto_stop: bool = False,
    ) -> None:
        """Constructor for object pointing to a docker container.

        Args:
            container_cfg (ContainerConfig, optional): The configuration describing the container parameters.
                Defaults to None.
            remove_on_stop (bool, optional): Whether to delete the container after it is stopped.
                Defaults to False.
            auto_stop (bool, optional): Flag indicating if the container should be automatically stopped
                at the end of the lifecycle. Defaults to False.
        """
        self.client = docker.from_env()

        if container_cfg is None:
            container_cfg = ContainerConfig()

        try:
            self.container = self.client.containers.get(container_cfg.name)

        except docker.errors.NotFound:
            logger.info(f"Starting container {container_cfg.name}")
            self.container = self.client.containers.run(
                image=container_cfg.image,
                name=container_cfg.name,
                hostname=container_cfg.hostname,
                ports=container_cfg.ports,
                environment=container_cfg.environment,
                volumes=container_cfg.volumes,
                command=container_cfg.command,
                security_opt=container_cfg.security_opt,
                healthcheck=container_cfg.healthcheck,
                network=container_cfg.network,
                remove=remove_on_stop,
                detach=True,  # enabled so we don't block on this
            )

        self.auto_stop = auto_stop

    def __del__(self) -> None:
        if self.auto_stop:
            try:
                self.container.stop()
            except Exception as exc:
                logger.warning(f"Failed to stop container: {exc}")

        self.client.close()
