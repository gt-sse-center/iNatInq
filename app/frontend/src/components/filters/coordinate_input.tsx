interface CoordinateInputProps {
  label: string;
  name: string;
  min: string;
  max: string;
  setMin(min: string): void;
  setMax(max: string): void;
}

export function CoordinateInput({
  label,
  name,
  min,
  max,
  setMin,
  setMax,
}: CoordinateInputProps) {
  return (
    <div className="rounded-lg me-4 mb-1">
      <label className="w-16 inline-block text-xs mx-2 font-medium">
        {label}:
      </label>
      <input
        type="number"
        value={min}
        name={`${name.toLowerCase()}Min`}
        onChange={(e) => setMin(e.currentTarget.value)}
        className="w-16 px-2 py-1 text-xs rounded-lg outline-none bg-inatwhite text-inattext"
        step="0.01" // Suggests to browsers/keyboards that decimal input is expected
        inputMode="decimal" // Hints mobile keyboards to show a decimal pad
      />

      <label className="text-xs mx-2">to</label>
      <input
        type="number"
        value={max}
        name={`${name.toLowerCase()}Max`}
        onChange={(e) => setMax(e.currentTarget.value)}
        className="w-16 px-1 py-1 text-xs rounded-lg outline-none bg-inatwhite text-inattext"
        step="0.01" // Suggests to browsers/keyboards that decimal input is expected
        inputMode="decimal" // Hints mobile keyboards to show a decimal pad
      />
    </div>
  );
}
