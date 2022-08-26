import "../styles/globals.css";
import type { AppProps } from "next/app";
import { HomeProvider } from "@/providers/index";

function MyApp({ Component, pageProps }: AppProps) {
  return (
    <HomeProvider>
      <Component {...pageProps} />
    </HomeProvider>
  );
}

export default MyApp;
