interface BadgeProps {
  children: React.ReactNode;
  className: string;
}

export function BadgeGreen({ children, className }: BadgeProps) {
  return (
    <div
      className={`inline-block px-4 py-0.5 rounded-full bg-inatgreen text-inatwhite text-sm font-medium ${className}`}
    >
      {children}
    </div>
  );
}
