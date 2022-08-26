import ClipLoader from "react-spinners/ClipLoader";
import { CSSProperties } from "react";

type LoadingProps = {
  loading: boolean;
};

const override: CSSProperties = {
  left: "50%",
  top: "50%",
  transform: "translate(-50%, -50%)",
  position: "absolute",
};

export const IsLoading = ({ loading }: LoadingProps) => {
  return (
    <div
      style={{
        height: "100%",
        width: "100%",
        zIndex: 999,
        background: "rgb(white)",
        overflow: "hidden",
        position: "absolute",
      }}
    >
      <ClipLoader
        color={"black"}
        loading={loading}
        cssOverride={override}
        size={100}
      />
    </div>
  );
};
