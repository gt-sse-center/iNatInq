/**
 * Utility functions.
 */

/**
 * Helper function to get the value of an input element (specified by `key`) in the form (via `event`).
 * @param event The form submit event.
 * @param key The key of the input element to query.
 * @param defaultValue Default value if the input element has no value.
 * @returns The value from the input element in the form.
 */
export function getInputElementValue(
  event: React.FormEvent<HTMLFormElement>,
  key: string,
  defaultValue: string
): string {
  const input = event.currentTarget.elements.namedItem(
    key
  ) as HTMLInputElement | null;
  const value = input?.value ?? defaultValue;
  return value;
}
