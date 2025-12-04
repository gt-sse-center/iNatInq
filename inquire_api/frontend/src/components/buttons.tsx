interface ResetButtonProps {
  onClick(): void;
}

export function ResetButton({ onClick }: ResetButtonProps) {
  return (
    <button
      className="text-xs px-2 py-1 underline rounded hover:bg-sky-300 flex items-center text-inatlink font-medium"
      onClick={onClick}
      type="button"
    >
      Reset
    </button>
  );
}

interface PaginationButtonProps {
  children: React.ReactNode;
  disabled: boolean;
  onClick(): void;
}

export function PaginationButton({
  children,
  disabled,
  onClick,
}: PaginationButtonProps) {
  return (
    <button
      disabled={disabled}
      onClick={onClick}
      className="px-4 py-1 mx-2 border border-inatborder text-inattext rounded-lg hover:text-inatwhite hover:bg-inatlinkhover disabled:text-inattext disabled:bg-inatgray disabled:border-inatborder"
    >
      {children}
    </button>
  );
}
