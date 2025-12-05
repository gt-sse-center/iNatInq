import type { DateRange } from "@/interfaces/date_range";
import { DatePickerInput } from "@mantine/dates";
import { FilterSection } from "../filters/filter_section";

interface DateRangeFilterProps {
  dateRangeFilter: DateRange;
  setDateRangeFilter(range: DateRange): void;
}

export function DateRangeFilter({
  dateRangeFilter,
  setDateRangeFilter,
}: DateRangeFilterProps) {
  return (
    <FilterSection
      title="Date Range"
      description="Search within the specified dates."
      filterActive={Object.values(dateRangeFilter).every((value) => {
        return value !== null;
      })}
      onReset={() => {
        setDateRangeFilter({
          startDate: null,
          endDate: null,
        });
      }}
    >
      <div className="flex">
        <DatePickerInput
          placeholder="Start date"
          value={dateRangeFilter.startDate}
          onChange={(startDateString: string | null) => {
            if (startDateString) {
              const startDate = new Date(startDateString);
              setDateRangeFilter({
                ...dateRangeFilter,
                startDate,
              });
            }
          }}
          className="mx-1"
        />
        <DatePickerInput
          placeholder="End date"
          value={dateRangeFilter.endDate}
          onChange={(endDateString: string | null) => {
            if (endDateString) {
              const endDate = new Date(endDateString);
              setDateRangeFilter({
                ...dateRangeFilter,
                endDate,
              });
            }
          }}
          className="mx-1"
        />
      </div>
    </FilterSection>
  );
}
