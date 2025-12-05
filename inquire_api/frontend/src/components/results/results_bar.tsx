import type { Image } from "@/interfaces/image";
import * as Icon from "react-bootstrap-icons";

interface ResultsBarProps {
  images: Image[];
  numImagesShown: number;
  setNumImagesShown(n: number): void;
  sliderValue: number;
  setSliderValue(n: number): void;
  maxSliderValue: number;
  showDetailedView: boolean;
  setShowDetailedView(b: boolean): void;
}

export function ResultsBar({
  images,
  numImagesShown,
  setNumImagesShown,
  sliderValue,
  setSliderValue,
  maxSliderValue,
  showDetailedView,
  setShowDetailedView,
}: ResultsBarProps) {
  return (
    <div
      className={`flex pt-2 pb-2 items-center bg-inatgray border border-inatborder ${
        images.length > 0 ? "" : "hidden"
      }`}
    >
      <div className="flex items-center">
        <h2 className="text-sm ms-6 font-normal">
          Retrieved {numImagesShown} Images
        </h2>
        <button
          className="shrink-0 flex items-center px-3 h-7 ms-2 text-xs
            rounded-lg border border-inatgreen
            text-inatwhite bg-inatgreen
            hover:text-inatwhite hover:bg-inatlinkhover"
          onClick={() => {
            setNumImagesShown(images.length);
          }}
        >
          <Icon.ArrowsExpand className="w-4 h-4 me-2" />
          Show All
        </button>
      </div>

      <div className="grow flex px-4 py-1 items-center rounded-lg">
        <Icon.GridFill className="w-4 h-4 me-1" />
        <p className="text-sm font-medium me-3">Grid Size</p>
        <input
          id="default-range"
          type="range"
          value={sliderValue}
          min={0}
          max={maxSliderValue}
          onChange={(e) => {
            setSliderValue(parseInt(e.currentTarget.value));
          }}
          className="w-32 h-2 rounded-lg appearance-none cursor-pointer dark:bg-inattext"
        />
      </div>

      <button
        className="shrink-0 flex items-center px-3 py-1 mx-4 text-xs
          rounded-lg border border-inatgreen
          text-inatwhite bg-inatgreen
          hover:text-inatwhite hover:bg-inatlinkhover"
        onClick={() => {
          setShowDetailedView(!showDetailedView);
        }}
      >
        <Icon.Image className="w-4 h-4 me-2" />
        Show/Hide Detail View
      </button>
    </div>
  );
}
