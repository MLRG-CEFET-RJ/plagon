import styles from "@/styles/Header.module.css";
import { IconContext } from "react-icons";
import { GiMagnifyingGlass } from "react-icons/gi";

interface HeaderProps {
  title?: string;
}

const ICON_SIZE = "5em";

const Header = ({ title = "Plagon" }: HeaderProps) => {
  return (
    <header className={styles.header}>
      <IconContext.Provider value={{ size: ICON_SIZE }}>
        <GiMagnifyingGlass />
      </IconContext.Provider>
      <h1>{title}</h1>
    </header>
  );
};

export { Header };
