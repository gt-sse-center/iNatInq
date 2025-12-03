import { DEFAULT_IMAGES_PER_PAGE, GRID_SIZE_RANGE } from "@/constants";
import type { Image } from "@/interfaces/image";
import { useState } from "react";
import { ResultsBar } from "./results_bar";
import { ImageResultsViewer } from "./results_viewer";

interface QueryResultsProps {
  images: Image[];
  numImagesShown: number;
  setNumImagesShown(n: number): void;
  hoveredImageIdx: number;
  setHoveredImageIdx(id: number): void;
}

export function QueryResults({
  images,
  numImagesShown,
  setNumImagesShown,
  hoveredImageIdx,
  setHoveredImageIdx,
}: QueryResultsProps) {
  const [showDetailedView, setShowDetailedView] = useState(true); // Show a side view with more information about the image
  const [sliderValue, setSliderValue] = useState(1);

  /**
   * Function to reduce the number of images shown.
   */
  function showLessImages(): void {
    setNumImagesShown(
      Math.max(
        DEFAULT_IMAGES_PER_PAGE *
          (Math.ceil(numImagesShown / DEFAULT_IMAGES_PER_PAGE) - 1),
        DEFAULT_IMAGES_PER_PAGE
      )
    );
  }

  /**
   * Function to increase the number of images shown.
   */
  function showMoreImages(): void {
    setNumImagesShown(
      Math.min(numImagesShown + DEFAULT_IMAGES_PER_PAGE, images.length)
    );
  }

  return (
    <div className="grow flex flex-row h-full overflow-hidden relative">
      <div className="grow flex flex-col">
        <ResultsBar
          images={images}
          numImagesShown={numImagesShown}
          setNumImagesShown={setNumImagesShown}
          sliderValue={sliderValue}
          setSliderValue={setSliderValue}
          maxSliderValue={GRID_SIZE_RANGE.length - 1}
          showDetailedView={showDetailedView}
          setShowDetailedView={setShowDetailedView}
        />
        <ImageResultsViewer
          images={images}
          numImagesShown={numImagesShown}
          gridSize={GRID_SIZE_RANGE[sliderValue]}
          hoveredImageIdx={hoveredImageIdx}
          showDetailedView={showDetailedView}
          onImageHover={(id: number) => {
            const imageIdx = images.findIndex((img: Image) => img.id == id);
            setHoveredImageIdx(imageIdx);
          }}
          onShowLessImages={showLessImages}
          onShowMoreImages={showMoreImages}
        />
      </div>
    </div>
  );
}
