import { createContext, Dispatch } from "react";

interface homeContextProps {
  errorMessages: string[] | undefined;
  file: File | undefined;
  setErrorMessages: Dispatch<string[] | undefined>;
  setFile: Dispatch<File | undefined>;
}

export const FileContext = createContext({} as homeContextProps);
