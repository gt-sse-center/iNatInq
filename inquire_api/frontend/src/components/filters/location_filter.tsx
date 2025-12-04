import type { GeoLocationRange } from "@/interfaces/location";
import { FilterSection } from "../filters/filter_section";
import { LatLongInputs } from "./latlong_inputs";

interface LocationFilterProps {
  locationFilter: GeoLocationRange;
  setLocationFilter(range: GeoLocationRange): void;
}

export function LocationFilter({
  locationFilter,
  setLocationFilter,
}: LocationFilterProps) {
  return (
    <FilterSection
      title="Location"
      description="Search for a location by its latitude and longitude (to 0.01 degrees)."
      filterActive={Object.values(locationFilter).every((value) => {
        return value.length > 0;
      })}
      onReset={() => {
        setLocationFilter({
          latitudeMin: "",
          latitudeMax: "",
          longitudeMin: "",
          longitudeMax: "",
        });
      }}
    >
      <LatLongInputs
        location={locationFilter}
        setLocation={setLocationFilter}
      />
    </FilterSection>
  );
}
