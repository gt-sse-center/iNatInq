import { useEffect, useRef, useState } from "react";

export function AutocompleteForm({ suggestions, values, onChange }) {
  const [value, setValue] = useState("");
  const [currentSuggestions, setCurrentSuggestions] = useState([]);
  const [activeIdx, setActiveIdx] = useState(-1);
  const [species, setSpecies] = useState("");
  const dropdown = useRef(null);

  useEffect(() => {
    onChange(species);
  }, [species]);

  function getSuggestionLongName(suggestion) {
    return suggestion["common_name"]
      ? `${suggestion["common_name"]} (${suggestion["name"]})`
      : suggestion["name"];
  }

  function onKeyDown(e) {
    if (e.key === "Enter") {
      e.preventDefault();
      if (activeIdx >= 0) {
        const suggestion = currentSuggestions[activeIdx];
        setValue(suggestion["name"]);
        setSpecies(suggestion["name"]);
        setCurrentSuggestions([]);
        setActiveIdx(-1);
      }
    } else if (e.key === "ArrowDown") {
      setActiveIdx(Math.min(activeIdx + 1, currentSuggestions.length - 1));
    } else if (e.key === "ArrowUp") {
      setActiveIdx(Math.max(activeIdx - 1, 0));
    }
  }

  function fetchSuggestions(query) {
    fetch(`/api/autocomplete?query=${encodeURIComponent(query)}`)
      .then((resp) => resp.text())
      .then((dataString) => {
        return JSON.parse(dataString.replace(/\bNaN\b/g, "null"));
      })
      .then((data) => {
        setActiveIdx(data.length ? 0 : -1);
        setCurrentSuggestions(data);
      })
      .catch((err) => console.error(err));
  }

  function showSuggestions() {
    if (species || !value.trim()) {
      setCurrentSuggestions([]);
      setActiveIdx(-1);
      return;
    }
    fetchSuggestions(value);
  }

  function onSuggestionClick(suggestion) {
    setValue(suggestion["name"]);
    setSpecies(suggestion["name"]);
    setCurrentSuggestions([]);
  }

  function onInputValueChange(e) {
    const newVal = e.currentTarget.value;
    setValue(newVal);
    if (newVal !== species) setSpecies("");
  }

  useEffect(() => {
    showSuggestions();
  }, [value]);

  // useEffect(() => {
  //   if (value && !species) {
  //     for (let i = 0; i < currentSuggestions.length; i++) {
  //       if (currentSuggestions[i]['name'] === value) {
  //         setSpecies(value);
  //         break;
  //       }
  //     }
  //   }
  // }, [currentSuggestions]);

  useEffect(() => {
    if (!currentSuggestions.length) return;
    function handleClick(event) {
      if (dropdown.current && !dropdown.current.contains(event.target)) {
        setCurrentSuggestions([]);
      }
    }
    window.addEventListener("click", handleClick);
    return () => window.removeEventListener("click", handleClick);
  }, [currentSuggestions]);

  return (
    <div className="grow flex items-center bg-white rounded-lg text-slate-800 px-1">
      <div className="h-full flex items-center ms-1 text-lg">
        {species ? (
          <Icon.CheckCircleFill className="shrink-0 w-5 h-5 text-green-500" />
        ) : (
          "🐶"
        )}
      </div>
      <input
        type="text"
        value={value}
        name="species_full_name"
        autoComplete="off"
        onFocus={showSuggestions}
        onChange={onInputValueChange}
        onKeyDown={onKeyDown}
        className={
          "grow px-2 py-1.5 me-2 text-xs outline-none " +
          (species
            ? " font-medium text-green-700"
            : " font-normal text-slate-800")
        }
        placeholder="Canis lupus"
      />
      <input type="hidden" name="species" value={species} />
      {currentSuggestions.length > 0 && (
        <div className="h-0">
          <div
            ref={dropdown}
            className="w-[480px] absolute top-0 mt-2 ms-2 left-72 z-10 rounded-lg overflow-hidden bg-white border border-slate-300 shadow-lg"
          >
            <div className="bg-slate-200 flex items-center">
              <div className="grow text-sm font-bold px-4 py-2">
                Suggestions
              </div>
              <div className="shrink-0 text-sm px-4 py-2">
                <span className="text-xs font-medium">
                  Use ↑↓ to navigate and [enter] to select
                </span>
              </div>
            </div>
            {currentSuggestions.map((suggestion, idx) => (
              <div
                key={idx}
                onClick={() => onSuggestionClick(suggestion)}
                className={
                  `px-4 py-2 cursor-pointer text-sm border-b border-b-slate-300 last:border-b-0 ` +
                  (idx === activeIdx
                    ? "bg-blue-500 text-white"
                    : "hover:bg-slate-100")
                }
              >
                {getSuggestionLongName(suggestion)}
                <span className="text-xs opacity-80 text-white font-medium ms-2 bg-slate-700 bg-opacity-80 px-1 py-0.5 rounded-lg">
                  {suggestion["rank"]}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
