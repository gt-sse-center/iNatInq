import { type Image } from "@/interfaces/image";

interface ImageResultProps {
  img: Image;
  active: boolean;
  gridSize: number;
  onHover(): void;
}

function ImageResult({ img, active, gridSize, onHover }: ImageResultProps) {
  return (
    <div
      className={
        `flex mx-1 my-2 bg-black overflow-hidden cursor-pointer hover:opacity-75
        border-2 border-inatborder ` +
        `${active ? "scale-105 p-0.5 opacity-80 " : ""}`
      }
      style={{
        width: `${gridSize ?? 10}rem`,
        height: `${gridSize ?? 10}rem`,
      }}
      onMouseEnter={onHover}
      id={`img-${img.id}`}
    >
      <img
        src={img.src}
        className="object-cover w-full h-full border-2 border-inatwhite"
      />
    </div>
  );
}

interface ImageResultGridProps {
  images: Image[];
  gridSize: number;
  hoveredImageIdx: number;
  onImageHover(id: number): void;
}

export function ImageResultGrid({
  images,
  gridSize,
  hoveredImageIdx,
  onImageHover,
}: ImageResultGridProps) {
  return (
    <div className="p-2 flex flex-wrap justify-evenly">
      {images.map((img: Image, key: number) => {
        return (
          <ImageResult
            key={img.id}
            img={img}
            active={key == hoveredImageIdx}
            gridSize={gridSize}
            onHover={() => {
              onImageHover(img.id);
            }}
          />
        );
      })}
    </div>
  );
}
