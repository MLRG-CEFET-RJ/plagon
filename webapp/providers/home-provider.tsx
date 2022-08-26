import { FileContext } from "@/contexts/index";
import { useState } from "react";

interface HomeProviderProps {
  children: JSX.Element | JSX.Element[];
}

export const HomeProvider = ({ children }: HomeProviderProps) => {
  const [errorMessages, setErrorMessages] = useState<string[] | undefined>(
    undefined
  );
  const [file, setFile] = useState<File | undefined>(undefined);

  return (
    <FileContext.Provider
      value={{
        errorMessages: errorMessages,
        file: file,
        setErrorMessages: setErrorMessages,
        setFile: setFile,
      }}
    >
      {children}
    </FileContext.Provider>
  );
};
