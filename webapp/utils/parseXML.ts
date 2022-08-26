import { XMLParser } from "fast-xml-parser";
import fs from "fs/promises";

export async function parseXML(path: string) {
  const options = {
    ignoreAttributes: false,
  };
  const parser = new XMLParser(options);

  const data = await fs.readFile(path, { encoding: "utf-8" });
  const jsonObj = parser.parse(data);

  return jsonObj;
}
