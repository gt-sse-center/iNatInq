import { FilterSection } from "./filter_section";

interface SpeciesFilterProps {
  speciesFilter: string;
  setSpeciesFilter(species: string): void;
}

export function SpeciesFilter({
  speciesFilter,
  setSpeciesFilter,
}: SpeciesFilterProps) {
  return (
    <FilterSection
      title="Species"
      description="Search for a species by its common name or scientific name."
      filterActive={speciesFilter.length > 0}
      onReset={() => {
        setSpeciesFilter("");
      }}
    >
      <div className="grow flex items-center bg-inatwhite rounded-lg text-inatgray px-1">
        <div className="h-full flex items-center ms-1 text-lg">🐶</div>
        <input
          type="text"
          value={speciesFilter}
          name="species"
          autoComplete="off"
          onChange={(e) => {
            const newSpecies = e.currentTarget.value;
            setSpeciesFilter(newSpecies);
          }}
          className={
            "grow px-2 py-1.5 me-2 text-xs outline-none  font-normal text-inattext"
          }
          placeholder="Canis lupus"
        />
      </div>
    </FilterSection>
  );
}
