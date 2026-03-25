from collections.abc import Iterable, Iterator
import logging
from typing_extensions import override
from foundation.dead_letter_queue.dlq_backend import DLQBackend

logger = logging.getLogger(__name__)


class E2EDLQBackend(DLQBackend):
    @property
    @override
    def name(self) -> str:
        return "e2e-test-backend"

    @override
    def insert(self, image_id: str, *, metadata: dict[str, object] | None = None) -> None:
        logger.info("Inserting %s into the DLQ!", image_id)

    @override
    def get_queue_contents(self) -> Iterator[str]:
        stubs = ["cat.png", "bird.png", "penguin.jpeg"]
        logger.info("Pulling contents (%s) of DLQ!", str(stubs))
        return iter(stubs)

    @override
    def delete(self, image_ids: Iterable[str]) -> None:
        for id in image_ids:
            logger.info("Deleting %s from the DLQ!", id)
