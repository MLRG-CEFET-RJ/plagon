import { FileRejection } from "react-dropzone";

export function getErrorMessages(errorCodes: Set<string>): string[] {
  const errorMessages: string[] = [];

  errorCodes.forEach((errorCode: string) => {
    let errorMessage =
      "Erro genérico. Abra uma issue reportando o caminho até o erro.";
    if (errorCode === "file-invalid-type") {
      errorMessage =
        "Formato errado de arquivo. Por gentileza, selecione um arquivo com a extensão .pdf or .txt";
    }
    if (errorCode === "too-many-files") {
      errorMessage =
        "Mais de um arquivo selecionado. Por gentileza, selecione apenas um arquivo.";
    }
    errorMessages.push(errorMessage);
  });

  return errorMessages;
}

export function getErrorCodesSet(fileRejections: FileRejection[]): Set<string> {
  const errorCodesSet = new Set<string>();

  fileRejections.forEach((fileRejection: FileRejection) => {
    fileRejection.errors.forEach((error: { code: string; message: string }) => {
      errorCodesSet.add(error.code);
    });
  });

  return errorCodesSet;
}
