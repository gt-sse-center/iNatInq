import type { GeoLocation } from "./location";

export interface Image {
  id: number;
  src: string;
  score: number;
  file_name: string;
  species: string;
  location: GeoLocation;
  observed_on: Date;
}
