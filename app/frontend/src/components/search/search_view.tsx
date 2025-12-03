import { API_URL, DEFAULT_IMAGES_PER_PAGE } from "@/constants";
import { type QueryApiResult } from "@/interfaces/api";
import { type Image } from "@interfaces/image";
import { useState } from "react";
import { FilterBar } from "./filter_bar";
import { SearchBar } from "./search_bar";

interface SearchViewProps {
  setImages(images: Image[]): void;
  setNumImagesShown(num_images: number): void;
  setHoveredImageIdx(image_idx: number): void;
}

export function SearchView({
  setImages,
  setNumImagesShown,
  setHoveredImageIdx,
}: SearchViewProps) {
  // Whether to show the sidebar with advanced filters
  const [showAdvancedFilters, setShowAdvancedFilters] = useState(false);
  // Record the values of the filters as specified by the user.
  const [advancedFilters, setAdvancedFilters] = useState({});

  /**
   * Function to submit the form data to the query API endpoint.
   * It parses the response then set the `images` state with the results.
   * @param formData The Search form data which is the user query plus the filter parameters.
   */
  async function submitQuery(formData: FormData): Promise<void> {
    formData.append("filters", JSON.stringify(advancedFilters));

    console.log(`submitted: ${JSON.stringify(Object.fromEntries(formData))}`);

    await fetch(`${API_URL}/query`, {
      method: "POST",
      body: formData,
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error(
            "Network response was not ok: " + response.statusText
          );
        }

        // Return the response data as a JSON object
        return response.json();
      })
      .then((data: QueryApiResult[]) => {
        const newImages: Image[] = data.map(
          (result: QueryApiResult): Image => ({
            id: result.id, // The 'photo_id'
            src: result.img_url,
            score: result.score,
            file_name: result.file_name,
            species: result.species,
            location: result.location,
            observed_on: new Date(result.observed_on),
          })
        );

        setImages(newImages);
        setHoveredImageIdx(-1);
        // Only show a small number of images so we can paginate
        setNumImagesShown(DEFAULT_IMAGES_PER_PAGE);
      })
      .catch((error: Error) => {
        console.error(`fetch error: ${error.message}`);
      });
  }

  return (
    <div className="flex bg-inatgray text-inattext">
      <div className="px-6 pt-1 pb-3">
        <SearchBar
          submitCallback={submitQuery}
          onShowFilters={() => {
            setShowAdvancedFilters(!showAdvancedFilters);
          }}
        />

        <FilterBar
          className={showAdvancedFilters ? "" : "hidden"}
          onFilterChange={(filters: any) => {
            setAdvancedFilters(filters);
          }}
        />
      </div>
    </div>
  );
}
