import { DEFAULT_IMAGES_PER_PAGE } from "@/constants";
import type { Image } from "@/interfaces/image";
import * as Icon from "react-bootstrap-icons";
import { PaginationButton } from "../buttons";
import { DetailedView } from "./detailed_view";
import { ImageResultGrid } from "./image_results";

interface ImageResultsViewerProps {
  images: Image[];
  numImagesShown: number;
  gridSize: number;
  hoveredImageIdx: number;
  showDetailedView: boolean;
  onImageHover(id: number): void;
  onShowLessImages(): void;
  onShowMoreImages(): void;
}

export function ImageResultsViewer({
  images,
  numImagesShown,
  gridSize,
  hoveredImageIdx,
  showDetailedView,
  onImageHover,
  onShowLessImages,
  onShowMoreImages,
}: ImageResultsViewerProps) {
  // If images are available then render them, else display waiting message.
  return images.length > 0 ? (
    <div className="grow flex overflow-hidden">
      <div className="grow basis-0 overflow-scroll h-full">
        <ImageResultGrid
          images={images.slice(0, numImagesShown)}
          gridSize={gridSize}
          hoveredImageIdx={hoveredImageIdx}
          onImageHover={onImageHover}
        />

        <div className="py-5 text-center bg-inatwhite border-t border-t-inatborder">
          <PaginationButton
            disabled={numImagesShown <= DEFAULT_IMAGES_PER_PAGE}
            onClick={onShowLessImages}
          >
            <Icon.ChevronContract className="w-4 h-4 me-2 inline" />
            Show Less
          </PaginationButton>

          <PaginationButton
            disabled={numImagesShown >= images.length}
            onClick={onShowMoreImages}
          >
            <Icon.Grid3x2GapFill className="w-4 h-4 me-2 inline" />
            Show More
          </PaginationButton>
        </div>
      </div>

      <div
        className={`${
          showDetailedView ? "" : "hidden"
        } grow basis-0 bg-inatwhite border-s border-inatborder`}
      >
        {hoveredImageIdx != null && hoveredImageIdx >= 0 && (
          <DetailedView
            image={images[hoveredImageIdx]}
            imageIdx={hoveredImageIdx}
            numImagesShown={numImagesShown}
          />
        )}
      </div>
    </div>
  ) : (
    <div className="grow flex items-center justify-center">
      <p className="text-inattext">No results to show.</p>
    </div>
  );
}
