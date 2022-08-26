import type { XMLJson } from "@/types/types";

export function getPlagiarisms(xmlJson: XMLJson) {
  return xmlJson.document.feature.slice(2);
}
