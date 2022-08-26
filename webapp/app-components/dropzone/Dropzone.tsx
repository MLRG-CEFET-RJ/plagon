import styles from "@/styles/Dropzone.module.css";
import { FileRejection, useDropzone } from "react-dropzone";
import { FileContext } from "@/contexts/index";
import { IconContext } from "react-icons";
import { ImFileText } from "react-icons/im";
import { getErrorMessages, getErrorCodesSet } from "./utils";
import { useCallback, useEffect, useContext, Fragment } from "react";

const Dropzone = () => {
  const { errorMessages, setErrorMessages, file, setFile } =
    useContext(FileContext);

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      setFile(acceptedFiles[0]);
    },
    [setFile]
  );

  const onDropRejected = useCallback(
    (fileRejections: FileRejection[]) => {
      const errorCodesSet = getErrorCodesSet(fileRejections);
      setErrorMessages(getErrorMessages(errorCodesSet));
    },
    [setErrorMessages]
  );

  useEffect(() => {
    if (errorMessages) {
      alert(errorMessages.join("\n"));
      setErrorMessages(undefined);
    }
  }, [errorMessages, setErrorMessages]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    onDropRejected,
    accept: {
      "text/html": [".pdf", ".txt"],
    },
    maxFiles: 1,
  });

  return (
    <div {...getRootProps()} className={styles.container}>
      <input {...getInputProps()} id="dropzone" />
      {file ? (
        <div className={styles.dropzoneTextContainer}>
          <IconContext.Provider value={{ size: "1.5rem" }}>
            <div className={styles.iconContainer}>
              <ImFileText />
            </div>
          </IconContext.Provider>
          <p className="text-bold">{file.name}</p>
        </div>
      ) : (
        <Fragment>
          {isDragActive ? (
            <p>Solte o arquivo aqui</p>
          ) : (
            <p>Arraste o arquivo, ou clique para selecionar</p>
          )}
        </Fragment>
      )}
    </div>
  );
};

export { Dropzone };
