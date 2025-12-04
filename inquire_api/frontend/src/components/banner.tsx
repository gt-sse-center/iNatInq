import inaturalistLogoUrl from "@/assets/inaturalist-logo.svg";

export function Banner() {
  return (
    <div
      id="inquire-banner"
      className="flex items-center bg-inatgray text-inattext grow px-6 py-2"
    >
      <div className="grow">
        <div className="flex">
          <a
            className="pt-1"
            href="https://www.inaturalist.org/"
            target="_blank"
          >
            <img src={inaturalistLogoUrl} className="w-32" />
          </a>
          <h1 className="grow ml-2 pl-2 text-2xl">
            INQUIRE: Search Natural World Images with Text
          </h1>
        </div>

        <p className="text-sm mt-2 mb-0">
          Enter the concepts you're interested in to find relevant iNaturalist
          photos with AI.
        </p>
      </div>
    </div>
  );
}
