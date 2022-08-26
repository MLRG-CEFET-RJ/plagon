import type { NextApiRequest, NextApiResponse } from "next";
import type { Plagiarism } from "@/types/types";
import { getPlagiarisms, parseXML } from "@/utils/index";
import path from "path";

type Data = {
  text: string;
  plagiarisms: Plagiarism[];
};

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<{ status: number; message: string; data?: Data }>
) {
  if (req.method === "POST") {
    const text = req.body;
    /*** SEND TEXT (text) TO THE NEURAL NET ***/
    /*** GET THE XML OUTPUTED FROM THE NEURAL NET AND SEND IT AS A RESPONSE ***/
    const exampleDir = path.join(process.cwd(), "examples");
    const xmlJson = await parseXML(
      `${exampleDir}/suspicious-document00001.xml`
    );
    const plagiarisms = getPlagiarisms(xmlJson);

    res.status(201).json({
      status: 201,
      message: "Text and plagiarisms received",
      data: { text: text, plagiarisms: plagiarisms },
    });
  } else {
    res
      .status(405)
      .send({ status: 405, message: "Only POST requests allowed" });
  }
}
