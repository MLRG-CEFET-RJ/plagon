export type Plagiarism = {
  "@_name": string;
  "@_this_offset": string;
  "@_this_length": string;
};

export type XMLJson = {
  "?xml": { "@_version": string; "@_encoding": string };
  document: { feature: Plagiarism[] };
  "@_xmlns:xsi": string;
  "@_xsi:noNamespaceSchemaLocation": string;
  "@_reference": string;
};
