import type { GeoLocation } from "./location";

/**
 * Interface defining the result of an API call to the query endpoint.
 */
export interface QueryApiResult {
  id: number;
  img_url: string;
  score: number;
  file_name: string;
  species: string;
  location: GeoLocation;
  observed_on: string;
}
