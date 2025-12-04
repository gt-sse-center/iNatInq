import type { GeoLocationRange } from "@interfaces/location";
import { CoordinateInput } from "./coordinate_input";

interface LatLongInputsProps {
  location: GeoLocationRange;
  setLocation(location: GeoLocationRange): void;
}

export function LatLongInputs({ location, setLocation }: LatLongInputsProps) {
  /**
   * Set an example latitude and longitude range.
   * This works well for the query: California condor with green '26' on its wing
   */
  function populateExample() {
    setLocation({
      latitudeMin: "32.33",
      latitudeMax: "42.38",
      longitudeMin: "-171.14",
      longitudeMax: "-101.01",
    });
  }

  return (
    <div className="">
      <CoordinateInput
        label="Latitude"
        name="Latitude"
        min={location.latitudeMin}
        max={location.latitudeMax}
        setMin={(value: string) => {
          const newLocation = structuredClone(location);
          newLocation.latitudeMin = value;
          setLocation(newLocation);
        }}
        setMax={(value: string) => {
          const newLocation = structuredClone(location);
          newLocation.latitudeMax = value;
          setLocation(newLocation);
        }}
      />
      <CoordinateInput
        label="Longitude"
        name="Longitude"
        min={location.longitudeMin}
        max={location.longitudeMax}
        setMin={(value: string) => {
          const newLocation = structuredClone(location);
          newLocation.longitudeMin = value;
          setLocation(newLocation);
        }}
        setMax={(value: string) => {
          const newLocation = structuredClone(location);
          newLocation.longitudeMax = value;
          setLocation(newLocation);
        }}
      />
      <button
        className="text-xs text-inattext underline hover:text-inatlinkhover"
        onClick={populateExample}
      >
        Use Example
      </button>
    </div>
  );
}
