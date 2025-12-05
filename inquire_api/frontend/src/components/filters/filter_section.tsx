import type React from "react";
import { BadgeGreen } from "../badge";
import { ResetButton } from "../buttons";

interface FilterSectionProps {
  children: React.ReactNode;
  title: string;
  description: string;
  filterActive: boolean;
  onReset(): void;
}

export function FilterSection({
  children,
  title,
  description,
  filterActive,
  onReset,
}: FilterSectionProps) {
  return (
    <div className="mr-4">
      <div className="flex items-center justify-between mb-1">
        <div className="text-md text-inattext font-bold">{title}</div>
        <ResetButton onClick={onReset} />
      </div>
      <div className="text-xs mt-2 mb-4">{description}</div>
      {children}
      <div className="text-xs mt-4 mx-1">
        {filterActive ? (
          <BadgeGreen className="text-xs mt-1">Filter is active</BadgeGreen>
        ) : (
          <p className="text-inattext italic">
            Filter is currently not set or not valid.
          </p>
        )}
      </div>
    </div>
  );
}
