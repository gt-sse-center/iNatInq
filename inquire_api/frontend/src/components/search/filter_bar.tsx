import { type GeoLocationRange } from "@interfaces/location";
import { useEffect, useState } from "react";

import { type DateRange } from "@/interfaces/date_range";
import { DateRangeFilter } from "../filters/date_range_filter";
import { LocationFilter } from "../filters/location_filter";
import { MonthFilter } from "../filters/month_filter";
import { SpeciesFilter } from "../filters/species_filter";

interface FilterBarProps {
  className?: string;
  onFilterChange(filters: any): void;
}

export function FilterBar({ className = "", onFilterChange }: FilterBarProps) {
  const [speciesFilter, setSpeciesFilter] = useState("");
  const [locationFilter, setLocationFilter] = useState<GeoLocationRange>({
    latitudeMin: "",
    latitudeMax: "",
    longitudeMin: "",
    longitudeMax: "",
  });
  const [dateRangeFilter, setDateRangeFilter] = useState<DateRange>({
    startDate: null as Date | null,
    endDate: null as Date | null,
  });
  const [monthsFilter, setMonthsFilter] = useState<string[]>([]);

  // If the filters are updated, then call `onFilterChange` to record it.
  useEffect(() => {
    const filters = {
      species: speciesFilter,
      location: locationFilter,
    };
    onFilterChange(filters);
  }, [speciesFilter, locationFilter]);

  return (
    <div className={`mt-4 ${className}`}>
      <div className="flex grid grid-cols-4">
        <SpeciesFilter
          speciesFilter={speciesFilter}
          setSpeciesFilter={setSpeciesFilter}
        />

        <LocationFilter
          locationFilter={locationFilter}
          setLocationFilter={setLocationFilter}
        />

        <DateRangeFilter
          dateRangeFilter={dateRangeFilter}
          setDateRangeFilter={setDateRangeFilter}
        />

        <MonthFilter
          monthsFilter={monthsFilter}
          setMonthsFilter={setMonthsFilter}
        />
      </div>
    </div>
  );
}
