import { DEFAULT_IMAGE_COUNT } from "@/constants";
import * as Icon from "react-bootstrap-icons";

interface SearchFormProps {
  formRef: React.RefObject<HTMLFormElement | null>;
  query: string;
  setQuery(q: string): void;
  loading: boolean;
  onSubmit(event: React.FormEvent<HTMLFormElement>): Promise<void>;
  onShowFilters(): void;
}

export default function SearchForm({
  formRef,
  query,
  setQuery,
  loading,
  onSubmit,
  onShowFilters,
}: SearchFormProps) {
  return (
    <form id="inquire-search" onSubmit={onSubmit} ref={formRef}>
      <div className="flex mt-4 h-10 items-center color-inattext">
        <div className="grow flex items-center bg-inatwhite rounded-lg overflow-hidden">
          <div className="h-full flex items-center px-1 ms-2">
            <Icon.Search className="shrink-0 w-5 h-5" />
          </div>
          <input
            type="text"
            value={query}
            name="user_input"
            onChange={(e) => {
              setQuery(e.currentTarget.value);
            }}
            className="grow px-3 py-2 text-md focus:border-inatborder"
            placeholder="Some search query..."
            maxLength={256}
          />
          <input type="hidden" name="k" value={DEFAULT_IMAGE_COUNT} />
        </div>
        <button
          disabled={query == ""}
          type="submit"
          className="h-full shrink-0 px-2 ms-2 flex items-center
              rounded-lg
              text-inatwhite
              bg-inatgreen border-inatgreen
              enabled:cursor-pointer
              disabled:text-inattext disabled:bg-inatgray 
              disabled:border disabled:border-inatborder"
        >
          <Icon.ArrowRightShort className="w-8 h-8" />
        </button>
        <div className={`ms-4 flex items-center ${loading ? "" : "hidden"}`}>
          <Icon.ArrowRepeat className="w-8 h-8 animate-spin" />
          <p className="ms-2">Loading...</p>
        </div>
        <button
          type="button"
          className="shrink-0 flex items-center rounded-lg text-sm text-inatwhite bg-inatlink hover:bg-inatlinkhover px-2 ms-3 h-8 cursor-pointer"
          onClick={onShowFilters}
        >
          <Icon.Filter className="w-5 h-5 inline me-1" /> Advanced Filters
        </button>
      </div>
    </form>
  );
}
