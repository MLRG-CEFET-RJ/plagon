import type { Plagiarism } from "@/types/types";

export function buildTextHTML(
  plagiarisms: Plagiarism[],
  text: string,
  style: string
) {
  let styledText = text;

  let charAddedCounter = 0;

  const openingTag = `<span className="${style}">`;
  const closeningTag = "</span>";

  plagiarisms.forEach((plagiarism) => {
    const plagStartIndex = parseInt(plagiarism["@_this_offset"]);
    const plagEndIndex = plagStartIndex + parseInt(plagiarism["@_this_length"]);

    const plagiarizedPart =
      openingTag + text.substring(plagStartIndex, plagEndIndex) + closeningTag;

    styledText =
      styledText.substring(0, plagStartIndex + charAddedCounter) +
      plagiarizedPart +
      styledText.substring(plagEndIndex + charAddedCounter);

    charAddedCounter += openingTag.length + closeningTag.length;
  });

  return styledText;
}
