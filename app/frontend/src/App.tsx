import { Banner } from "@/components/banner";
import { QueryResults } from "@/components/results/query_results";
import { SearchView } from "@/components/search/search_view";
import { DEFAULT_IMAGES_PER_PAGE } from "@/constants";
import type { Image } from "@/interfaces/image";
import { MantineProvider } from "@mantine/core";
import { useEffect, useState } from "react";

import "@mantine/core/styles.css";
import "@mantine/dates/styles.css";
import "./App.css";

function App() {
  // resulting images from the query
  const [images, setImages] = useState<Array<Image>>([]);
  // The index of the image being hovered over.
  const [hoveredImageIdx, setHoveredImageIdx] = useState(-1);
  // The number of images shown in the QueryResults
  const [numImagesShown, setNumImagesShown] = useState(DEFAULT_IMAGES_PER_PAGE);

  //FIXME(Varun)
  // const [mapSpeciesToCommonName, setMapSpeciesToCommonName] = useState({});
  // const loadSpeciesNames = async () => {
  //   const response = await fetch(`/map_species_to_common_name.json`);
  //   if (!response.ok) {
  //     throw new Error(`HTTP Error! status: ${response.status}`);
  //   }
  //   const entries = await response.json();
  //   setMapSpeciesToCommonName(entries);
  // };

  // // On initial page load, load species common names
  // useEffect(() => {
  //   loadSpeciesNames().catch((e) => {
  //     // console.error("An error occured while fetching species data: ", e);
  //   });
  // }, []);

  /**
   * Helper function to scroll to the highlighted Image `img` in the QueryResults.
   * @param img The image which was is highlighted.
   * @returns None
   */
  function scrollToImage(img: Image | null | undefined): void {
    if (!img) return;
    const element = document.getElementById(`im-${img.id}`);
    if (!element) return;
    element?.scrollIntoView({
      behavior: "smooth",
      block: "center",
      inline: "nearest",
    });
  }

  /**
   * Effect which handles arrow key presses to move
   * the highlighted image in the image results viewer.
   */
  useEffect(() => {
    const keyDownHandler = (e: KeyboardEvent): void => {
      const formElements = ["INPUT", "TEXTAREA", "SELECT", "OPTION"];

      // Ignore key presses within form elements, like inputs
      const targetElement = e.target as HTMLElement;
      if (formElements.includes(targetElement.tagName) || images.length == 0) {
        // Do nothing
        return;
      }

      e.preventDefault();
      let newIdx: number = hoveredImageIdx;
      if (e.key == "ArrowRight") {
        // Move to the next image
        newIdx =
          hoveredImageIdx >= 0
            ? Math.min(hoveredImageIdx + 1, numImagesShown - 1)
            : hoveredImageIdx;
      } else if (e.key == "ArrowLeft") {
        // Move to the previous image
        newIdx =
          hoveredImageIdx >= 0
            ? Math.max(hoveredImageIdx - 1, 0)
            : hoveredImageIdx;
      }

      setHoveredImageIdx(newIdx);
      scrollToImage(images[newIdx]);
    };
    document.addEventListener("keydown", keyDownHandler);

    // clean up
    return () => {
      document.removeEventListener("keydown", keyDownHandler);
    };
  }, [hoveredImageIdx, numImagesShown]);

  return (
    <MantineProvider>
      <div className="flex flex-col h-[100vh] w-full min-w-min">
        {/* <TermsModal /> */}

        {/* Top banner */}
        <Banner />

        {/* The Search Bar */}
        <SearchView
          setImages={setImages}
          setNumImagesShown={setNumImagesShown}
          setHoveredImageIdx={setHoveredImageIdx}
        />

        {/* The viewer with all the query results */}
        <QueryResults
          images={images}
          numImagesShown={numImagesShown}
          setNumImagesShown={setNumImagesShown}
          hoveredImageIdx={hoveredImageIdx}
          setHoveredImageIdx={setHoveredImageIdx}
        />
      </div>
    </MantineProvider>
  );
}

export default App;
