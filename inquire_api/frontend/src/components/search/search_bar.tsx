import { useRef, useState } from "react";
import ExampleQueries from "./example_queries";
import SearchForm from "./search_form";

interface SearchBarProps {
  submitCallback(formData: FormData): Promise<void>;
  onShowFilters(): void;
}

export function SearchBar({ submitCallback, onShowFilters }: SearchBarProps) {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);

  // Handy reference to the form for use in ExampleQueries.
  const formRef = useRef<HTMLFormElement>(null);

  async function onSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    setLoading(true);
    await submitCallback(new FormData(event.currentTarget));
    setLoading(false);
  }

  return (
    <div>
      <ExampleQueries
        formRef={formRef}
        setQuery={setQuery}
        loading={loading}
        setLoading={setLoading}
        submitCallback={submitCallback}
      />

      <SearchForm
        formRef={formRef}
        query={query}
        setQuery={setQuery}
        loading={loading}
        onSubmit={onSubmit}
        onShowFilters={onShowFilters}
      />
    </div>
  );
}
