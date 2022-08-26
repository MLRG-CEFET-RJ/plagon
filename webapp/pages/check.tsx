import type { Plagiarism } from "@/types/types";

import Link from "next/link";
import Router from "next/router";
import parse from "html-react-parser";
import styles from "@/styles/Check.module.css";
import { Header, IsLoading } from "@/components/index";
import { FileContext } from "@/contexts/index";
import { NextPage } from "next";
import { buildTextHTML } from "../utils/buildTextHtml";
import { Fragment, useContext, useEffect, useState } from "react";

const Check: NextPage = () => {
  const [text, setText] = useState<string>("");
  const [plagiarisms, setPlagiarisms] = useState<Plagiarism[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  const { file, setFile } = useContext(FileContext);

  useEffect(() => {
    (async () => {
      try {
        const response = await fetch("/api/check", {
          method: "POST",
          body: file,
        });

        const responseJson = await response.json();
        setText(responseJson.data.text);
        setPlagiarisms(responseJson.data.plagiarisms);

        setIsLoading(false);
      } catch (error) {
        alert(`The following error ocurred: ${JSON.stringify(error)}`);
        Router.push("/");
      }
    })();

    return () => {
      setIsLoading(false);
    };
  }, [file, setFile]);

  const phrasesCounter = text.split(".").length;
  const plagiarismCounter = plagiarisms.length;

  let styledText = "";
  if (plagiarismCounter > 0) {
    styledText = buildTextHTML(plagiarisms, text, `${styles.plagiarizedPart}`);
  }

  return (
    <Fragment>
      {isLoading ? (
        <IsLoading loading={isLoading} />
      ) : (
        <div className={styles.container}>
          <Header />
          <main className={styles.mainContainer}>
            <section className={styles.titlesContainer}>
              <h1 className="text-bold">Relatório de Varredura de plágio</h1>
              <div className={styles.subtitlesContainer}>
                <h3>
                  <span className="text-bold">Frases: </span>
                  {phrasesCounter}
                </h3>
                <h3>
                  <span className="text-bold">Possíveis plágios: </span>
                  {plagiarismCounter}
                </h3>
              </div>
            </section>
            <section className={styles.textArea}>
              <article>{parse(styledText)}</article>
            </section>
            <div className={styles.buttonContainer}>
              <Link href="/">
                <button
                  className="secondary-button"
                  onClick={() => setFile(undefined)}
                >
                  Inicar nova pesquisa
                </button>
              </Link>
            </div>
          </main>
        </div>
      )}
    </Fragment>
  );
};

export default Check;
