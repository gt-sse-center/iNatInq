import { type Image } from "@/interfaces/image";
import * as Icon from "react-bootstrap-icons";

interface DetailedViewProps {
  image: Image;
  imageIdx: number;
  numImagesShown: number;
}

export function DetailedView({
  image,
  imageIdx,
  numImagesShown,
}: DetailedViewProps) {
  return (
    <div className="w-full h-full text-center flex flex-col">
      <div className="grow min-h-0">
        <img src={image.src} className="object-contain w-full h-full p-4" />
      </div>
      <div className="shrink-0 pb-1 text-inattext text-sm">
        Image {imageIdx + 1} of {numImagesShown}
      </div>
      <div
        className="shrink-0 flex items-center justify-center px-4 py-4
          text-md text-inattext font-medium
          bg-inatgray border-t border-t-inatborder"
      >
        <span className="mx-1 px-1 py-1 text-sm">
          Species: <i className="font-bold">{image.species}</i>
        </span>
        <span className="mx-1 px-1 py-1 text-sm">
          Observed: <b>{image.observed_on.toDateString()}</b>{" "}
          [<a className="text-inatlink"
            href={`https://maps.google.com/?q=${image.location.lat},${image.location.lon}`}
            target="_blank"
          >
            location
          </a>]
        </span>
        {image.id && (
          <a
            className="flex items-center mx-4 px-3 py-1
            underline text-sm
            border border-inatborder rounded-lg
            hover:text-inatwhite hover:bg-inatlinkhover"
            href={`https://www.inaturalist.org/photos/${image.id}`}
            target="_blank"
          >
            Open in iNaturalist
            <Icon.BoxArrowUpRight className="inline-block w-4 h-4 ms-1" />
          </a>
        )}
      </div>
    </div>
  );
}
