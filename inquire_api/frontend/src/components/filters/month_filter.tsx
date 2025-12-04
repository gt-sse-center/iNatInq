import { Chip } from "@mantine/core";
import { FilterSection } from "./filter_section";

interface MonthFilterProps {
  monthsFilter: string[];
  setMonthsFilter(months: string[]): void;
}

export function MonthFilter({
  monthsFilter,
  setMonthsFilter,
}: MonthFilterProps) {
  const months: string[] = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
  ];

  const monthChips = months.map((month) => (
    <Chip key={month} value={month} radius="xs">
      {month}
    </Chip>
  ));

  return (
    <FilterSection
      title="Month Selector"
      description="Select the months to filter on."
      filterActive={monthsFilter.length > 0}
      onReset={() => {
        setMonthsFilter([]);
      }}
    >
      <div className="grid grid-cols-3 gap-2">
        <Chip.Group
          multiple
          value={monthsFilter}
          onChange={(value: string[]) => {
            console.log(value);
            setMonthsFilter(value);
          }}
        >
          {monthChips}
        </Chip.Group>
      </div>
    </FilterSection>
  );
}
