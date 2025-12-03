interface ExampleQueriesProps {
  formRef: React.RefObject<HTMLFormElement | null>;
  setQuery(q: string): void;
  loading: boolean;
  setLoading(b: boolean): void;
  submitCallback(formData: FormData): Promise<void>;
}

export default function ExampleQueries({
  formRef,
  setQuery,
  setLoading,
  submitCallback,
}: ExampleQueriesProps) {
  const exampleQueries = [
    "California condor with green '26' on its wing",
    // 'Tagged swan',
    "A close-up of a whale fluke",
    // 'Black Skimmer performing skimming',
    // 'Honey Bee carrying pollen baskets',
    "hermit crab using plastic waste as its shell",
    // 'Satin bowerbird in its ornamented bower',
    // 'Fishing net on a reef',
    // 'bullhead shark egg case',
    "measuring the body dimensions of a bee",
    "silly cat in a hat",
  ];

  async function onClick(q: string): Promise<void> {
    // Set the query term in the main form.
    await setQuery(q);

    // Convert the form (using the formRef) to FormData and submit
    if (formRef.current) {
      const formData = new FormData(formRef.current);
      setLoading(true);
      await submitCallback(formData);
      setLoading(false);
    }
  }

  return (
    <div className="mt-0">
      <span className="me-0 text-base text-inattext">Examples: </span>
      {exampleQueries.map((q, idx) => (
        <button
          key={idx}
          className="border border-inatgreen rounded-full
              text-sm text-inatwhite
              bg-inatgreen
              px-3 py-1 mx-1 mt-2
              hover:opacity-80 cursor-pointer"
          onClick={() => onClick(q)}
        >
          {q}
        </button>
      ))}
    </div>
  );
}
